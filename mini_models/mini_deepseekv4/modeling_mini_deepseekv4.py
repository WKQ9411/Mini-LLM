import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from ..attention import DeepSeekV4Attention
from ..norm import RMSNorm
from ..cache import MiniDeepSeekV4CacheLayer
from ..rope import RotaryEmbedding
from .configuration_mini_deepseekv4 import MiniDeepSeekV4Config

# 参考代码:
# - huggingface 官方 inference 源码: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py
# - transformers 源码: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py


@dataclass
class MiniDeepSeekV4ModelOutput(BaseModelOutputWithPast):
    hidden_states_for_mtp: Optional[torch.FloatTensor] = None
    total_seq_aux_loss: Optional[torch.Tensor] = None
    all_global_counts: Optional[list[dict]] = None


@dataclass
class MiniDeepSeekV4ForCausalLMOutput(CausalLMOutputWithPast):
    total_seq_aux_loss: Optional[torch.Tensor] = None
    total_mtp_loss: Optional[torch.Tensor] = None
    all_global_counts: Optional[list[dict]] = None


class MiniDeepSeekV4Expert(nn.Module):
    """mini_deepseekv4 专家网络, 结构为带可选 clamp 的 SwiGLU"""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate = self.w1(hidden_states)
        up = self.w3(hidden_states)
        
        # 按照论文中的描述，将 SwiGLU 的线性分量限制在 [-10,10] 内，将门控分量上界为设置为 10
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        hidden_states = F.silu(gate) * up
        
        # 这里的 weights 是专家的路由权重
        if weights is not None:
            hidden_states = weights * hidden_states
        return self.w2(hidden_states)


class MiniDeepSeekV4Gate(nn.Module):
    """
    即 Router, MoE 中的门控网络, 用于动态路由, 支持 hash-based routing 和 top-k routing 两种方式
    其中, hash-based routing 是通过对 input_ids 进行哈希得到专家索引, 是预先确定的, 仅对前 n_hash_layers 层启用
    整体过程是: 
     1. 对专家进行分组, 共 n_groups 个组
     2. 每个组计算 2 个最大亲和度得分之和
     3. 根据上述结果选出 topk_groups 个组
     4. 从上述 topk_groups 个组的所有专家中, 选出 topk 个专家
    """

    def __init__(self, layer_idx: int, config: MiniDeepSeekV4Config):
        super().__init__()
        self.config = config
        self.topk = config.n_activated_experts
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.hash = layer_idx < config.n_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        if self.hash:
            # 自定义哈希映射
            tid2eid = self._build_tid2eid(
                vocab_size=config.vocab_size,
                num_experts=config.n_routed_experts,
                topk=config.n_activated_experts,
                seed=42,
            )
            self.register_buffer("tid2eid", tid2eid, persistent=True)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(config.n_routed_experts), requires_grad=False)  # 用于无辅助损失负载均衡策略的 bias，不参与梯度计算，基于策略来更新bias，可以理解为通过策略干预而不是 loss 来进行更新的模型参数
        self.use_noaux_load_balance = config.use_noaux_load_balance
        self.original_scores: Optional[torch.Tensor] = None  # 用于存储原始的亲和度得分, 形状为 (batch_size * seq_len, n_routed_experts)

    def _build_tid2eid(
        self,
        vocab_size: int,
        num_experts: int,
        topk: int,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        一个简单的均匀 hash routing 表生成方式, 根据 token_id 为每个 token 分配 topk 个专家, 以实现负载均衡的 hash routing
        更理想的方式应当是根据每个 token_id 在预料中的分布情况来实现负载均衡

        思路：
        - 按 token_id 顺序逐个分配
        - 每次优先选择当前全局负载最低的专家
        - 如果多个专家负载相同, 随机打乱后选 topk
        - 保证同一个 token 的 topk experts 不重复
        """
        g = torch.Generator()
        g.manual_seed(seed)

        tid2eid = torch.empty(vocab_size, topk, dtype=torch.long)  # (vocab_size, topk)
        expert_load = torch.zeros(num_experts, dtype=torch.long)  # (num_experts,) 记录每个专家当前的负载情况

        for tid in range(vocab_size):
            # 给专家加一点随机扰动，用来打破平衡点情况
            noise = torch.rand(num_experts, generator=g)

            # 优先选负载低的专家，负载相同则由 noise 决定
            score = expert_load.float() + noise * 1e-3
            experts = torch.argsort(score)[:topk]

            tid2eid[tid] = experts
            expert_load[experts] += 1

        return tid2eid
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        门控网络的前向传播

        Args:
            hidden_states (torch.Tensor): 输入按 token 排列, 形状为 (batch_size * seq_len, hidden_size), 在输入前已经调整好形状
            input_ids (torch.Tensor): 输入 token ids, 形状为 (batch_size * seq_len), 用于 hash MoE

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 路由权重和选择的专家索引, 形状均为 (batch_size * seq_len, topk)
        """
        scores = F.linear(hidden_states, self.weight)  # 计算所有 token 对专家的亲和度得分 (batch_size * seq_len, n_routed_experts)
        if self.score_func == "softmax":  # DeepSeek-V2 
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":  # DeepSeek-V3
            scores = scores.sigmoid()
        elif self.score_func == "sqrtsoftplus":  # DeepSeek-V4
            scores = F.softplus(scores).sqrt()
        else:
            raise ValueError(f"Unsupported MoE score_func: {self.score_func}")
        self.original_scores = scores  # 保留原始得分，用于后续根据原始得分抽取 topk 个专家和序列级辅助损失计算 (batch_size * seq_len, n_routed_experts)
        
        if self.config.use_noaux_load_balance and self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]  # (batch_size * seq_len, topk)
        else:
            indices = scores.topk(self.topk, dim=-1)[1]  # DeepSeekV4 取消了对路由目标节点数量的约束，这里直接 topk，不再给专家分组 (batch_size * seq_len, topk)
        
        weights = self.original_scores.gather(dim=1, index=indices)  # 从原始分数中按选出的索引抽取出亲和度得分，即权重 (batch_size * seq_len, topk)
        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)  # 将权重归一化【注意，这是在选出的 topk 中进行归一化】
        weights = weights * self.route_scale  # 应用缩放因子
        return weights.type_as(hidden_states), indices


class MiniDeepSeekV4MoE(nn.Module):
    """Mixture-of-Experts (MoE) 混合专家模块, gate 将每个 token 路由到 top-k 个路由专家和 1 个共享专家"""

    def __init__(self, layer_idx: int, config: MiniDeepSeekV4Config):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        
        # 负载均衡配置
        self.use_seq_aux = config.use_seq_aux
        self.seq_aux_alpha = config.seq_aux_alpha
        self.bias_update_speed = config.bias_update_speed  # 用于无辅助损失负载均衡策略的 bias 的更新速度
        
        self.gate = MiniDeepSeekV4Gate(layer_idx, config)
        self.experts = nn.ModuleList(
            [
                MiniDeepSeekV4Expert(config.hidden_size, config.moe_intermediate_size, config.swiglu_limit)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.shared_experts = MiniDeepSeekV4Expert(
            config.hidden_size,
            config.n_shared_experts * config.moe_intermediate_size,
            config.swiglu_limit,
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        MoE 前向传播

        Args:
            hidden_states (torch.Tensor): 输入张量 (batch_size, seq_len, hidden_size)
            input_ids (torch.Tensor): 输入 token ids (batch_size, seq_len), 用于 hash MoE

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: 输出张量 (batch_size, seq_len, hidden_size), 本层序列级辅助损失, 本层全局负载情况
        """
        shape = hidden_states.shape
        batch_size, seq_len = shape[:2]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # 重新划分形状为 (batch_size * seq_len, hidden_size)

        weights, indices = self.gate(hidden_states, input_ids.flatten())  # 计算得到每个 token 的路由权重和选择的专家索引，形状均为 (batch_size * seq_len, topk)
        routed_output = torch.zeros_like(hidden_states)  # 用于累加路由专家的输出，形状为 (batch_size * seq_len, hidden_size)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)  # bincount 用于计算非负整数张量中每个值的出现次数，即此列表保存了一个 batch 里每个专家对应的激活次数，counts 的形状为 (n_routed_experts,)
        
        global_counts = counts.clone()
        if dist.is_available() and dist.is_initialized():
            # 同步所有 GPU 的 counts
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)

        # -------------------- 计算专家输出 --------------------
        # 为每个 token 计算路由专家的输出和
        for i in range(self.n_routed_experts):
            if counts[i] == 0:  # 如果这个 batch 中该专家没有被激活过，则跳过计算
                continue
            expert = self.experts[i]
            # 找到激活了第 i 个专家的 token，token_idx 代表行索引(即第几个 token)，top_idx 代表列索引(即该 token 的 top 几选择)，token_idx 和 top_idx 的类型为 torch.Tensor
            token_idx, top_idx = torch.where(indices == i)
            # 假设 n_matches 是匹配当前专家的 token 数量，那么 hidden_states[token_idx] 的形状是 (n_matches, hidden_size)，weights[token_idx, top_idx] 的形状是 (n_matches,)
            # None 用于增加一个维度，使形状变为 (n_matches, 1)
            routed_output[token_idx] += expert(hidden_states[token_idx], weights[token_idx, top_idx, None])

        # 计算共享专家的输出
        shared_output = self.shared_experts(hidden_states)  # 形状为 (batch_size * seq_len, hidden_size)
        output = (routed_output + shared_output).view(shape)  # 恢复形状 (batch_size, seq_len, hidden_size)

        # -------------------- 无辅助损失负载均衡策略 --------------------
        # 这里我们计算出了一个 batch 中所有专家的激活情况，故顺便在此应用无辅助损失的负载均衡策略来更新 gate 中的 bias
        # 每一个 MoE 层更新自己的 Gate 的 bias，下一个 batch 的数据将使用更新的 bias，训练的最后一组数据更新完 bias 后，将作为模型参数保存下来
        # 对于 hash routing 的 MoE 不使用无辅助损失的负载均衡策略
        if self.config.use_noaux_load_balance and self.training and not self.gate.hash:
            avg_count = sum(global_counts).float() / self.n_routed_experts  # 计算所有专家的平均激活次数
            
            # 仅 DDP 主进程和单卡时计算并更新 bias
            is_distributed_and_master = dist.is_initialized() and dist.get_rank() == 0
            is_not_distributed = not dist.is_initialized()
            if is_distributed_and_master or is_not_distributed:
                for i, count in enumerate(global_counts):
                    error = avg_count - count  # 计算每个专家的激活次数与平均激活次数的误差
                    self.gate.bias.data[i] += self.bias_update_speed * torch.sign(error)  # 应用无辅助损失的负载均衡策略来更新 bias
            
            # 广播更新后的 bias 到所有 GPU
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(self.gate.bias.data, src=0)
        
        # -------------------- 序列级别的辅助损失 --------------------
        # 如果使用了无辅助损失的负载均衡策略，那么计算序列级别的辅助损失时，使用未加 bias 的得分来计算 P_i，因为这体现的是 token 与专家真实的亲和度
        # 在 Gate 中也是类似的，bias 只是影响专家的选择，但最终的门控权重使用的是原始的真实亲和度得分，而不是加了 bias 的
        # 在计算 f_i 时，则使用的是实际激活的情况，这里实际上的激活情况是受 bias 影响的
        # 对于 hash routing 的 MoE 不使用序列级负载均衡损失
        seq_aux_loss = None
        if self.config.use_seq_aux and self.training and self.gate.original_scores is not None and not self.gate.hash:
            # 计算 P_i，含义为第 i 个专家在每个 token 上的平均归一化亲和度得分
            scores_for_seq_aux = self.gate.original_scores.view(batch_size, seq_len, -1)  # 此即原始的 s_{i,t} (batch_size, seq_len, n_routed_experts)
            scores_for_seq_aux = scores_for_seq_aux / scores_for_seq_aux.sum(dim=-1, keepdim=True)  # 沿着 n_routed_experts 的方向归一化，形成 s_{i,t}'
            p_i = scores_for_seq_aux.mean(dim=1)  # 沿着 token 的方向求平均 (batch_size, n_routed_experts)
            
            # 计算 f_i，含义为第 i 个专家在每个 token 上的平均激活次数
            # indices 计算了一个 batch 中每个 token 激活了哪些专家，现在要计算每个序列中，每个专家被哪些 token 激活
            # 可以使用 one-hot 编码来快速计算每个专家被多少个 token 激活
            f_i = F.one_hot(indices.view(batch_size, -1), num_classes=self.n_routed_experts)  # (batch_size, seq_len * topk, n_routed_experts)
            f_i = f_i.sum(dim=1)  # 沿 seq_len * topk 维度相加后，求出每个专家被多少个 token 激活 (batch_size, n_routed_experts)
            f_i = (f_i * self.n_routed_experts) / (self.n_activated_experts * seq_len)  # 计算每个专家的平均激活次数并乘以系数 (batch_size, n_routed_experts)
            seq_aux_loss = (f_i * p_i).sum() * self.seq_aux_alpha  # 计算序列级别的辅助损失

        return output, seq_aux_loss, global_counts


class MiniDeepSeekV4DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, config: MiniDeepSeekV4Config):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.layer_type = config.layer_types[layer_idx]
        self.rope_theta = config.rope_theta if self.layer_type == "sliding_attention" else config.compress_rope_theta
        self.rotary_emb = RotaryEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.rope_head_dim,
            rope_theta=self.rope_theta,
        )  # 纯 sliding 层使用 10000 rope_theta，HCA 和 CSA 均统一使用 40000

        self.attn = DeepSeekV4Attention(
            layer_idx=layer_idx,
            layer_types=config.layer_types,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            index_num_attention_heads=config.index_num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            o_lora_rank=config.o_lora_rank,
            head_dim=config.head_dim,
            index_head_dim=config.index_head_dim,
            rope_head_dim=config.rope_head_dim,
            compress_rope_theta=self.rope_theta,
            o_groups=config.o_groups,
            window_size=config.window_size,
            compress_ratios=config.compress_ratios,
            rms_norm_eps=config.rms_norm_eps,
            index_topk=config.index_topk,
            index_score_bias_alpha=config.index_score_bias_alpha,
            max_seq_len=config.max_position_embeddings,
        )
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = MiniDeepSeekV4MoE(layer_idx, config)

        # W_pre/W_post 将 n_hc * d 映射为 n_hc
        # W_res 将 n_hc * d 映射为 n_hc * n_hc
        # 因此设置 mHC 混合映射维度为 2 * n_hc + n_hc * n_hc，即 (2 + n_hc) * n_hc，其中论文中的 n_hc 就是代码中的 hc_mult
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size  # 即论文中的 n_hc * d
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))  # 投影矩阵
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))  # 投影矩阵
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))  # 偏置项
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))  # 偏置项
        self.hc_attn_scale = nn.Parameter(torch.empty(3))  # 门控因子
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))  # 门控因子

    def hc_pre(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完成 mHC 的动态参数化, 形成 A B C 矩阵, 同时映射形成输入 hidden_states
        
        Args:
            hidden_states: (batch_size, seq_len, hc_mult, hidden_size)
            hc_fn: (mix_hc, hc_mult * hidden_size)
            hc_scale: (3,)
            hc_base: (mix_hc,)
        
        Returns:
            reduced: (batch_size, seq_len, hidden_size) 经过输入映射 A 投影, 最终输入 attention 或 ffn 的 hidden states
            post: (batch_size, seq_len, hc_mult) 输出映射 C
            comb: (batch_size, seq_len, hc_mult, hc_mult) 残差映射 B
        """
        shape = hidden_states.shape
        dtype = hidden_states.dtype
        
        # 展平并归一化 (batch_size, seq_len, hc_mult * hidden_size)
        flat_states = hidden_states.flatten(2)
        rsqrt = torch.rsqrt(flat_states.float().square().mean(dim=-1, keepdim=True) + self.rms_norm_eps)  # RMSNorm
        mixes = F.linear(flat_states, hc_fn).float() * rsqrt  # 混合映射 (batch_size, seq_len, mix_hc)
        
        # 动态参数化
        pre, post, comb = self.hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        
        # pre (batch_size, seq_len, hc_mult) -> (batch_size, seq_len, hc_mult, 1)
        # hidden_states (batch_size, seq_len, hc_mult, hidden_size)
        # 沿 hc_mult 维度加权求和，得到 (batch_size, seq_len, hidden_size)
        reduced = torch.sum(pre.to(dtype).unsqueeze(-1) * flat_states.view(shape), dim=2)
        return reduced, post.to(dtype), comb.to(dtype)

    def hc_post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """
        进行输出映射, 并于残差流相加
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) attention 或 ffn 的输出
            residual: (batch_size, seq_len, hc_mult, hidden_size) 残差流
            post: (batch_size, seq_len, hc_mult) 输出映射 C
            comb: (batch_size, seq_len, hc_mult, hc_mult) 残差映射 B
        
        Returns:
            output: (batch_size, seq_len, hc_mult, hidden_size) 输出结果
        """
        output = post.unsqueeze(-1) * hidden_states.unsqueeze(-2)  # (batch_size, seq_len, hc_mult, hidden_size)
        output = output + torch.matmul(comb, residual)  # (batch_size, seq_len, hc_mult, hidden_size)
        return output.type_as(hidden_states)

    def hc_split_sinkhorn(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ):
        """
        mHC 动态参数化, 其中 mix_hc = (2 + hc_mult) * hc_mult

        Args:
            mixes: (batch_size, seq_len, mix_hc)
            hc_scale: (3,)
            hc_base: (mix_hc,)
            hc_mult: 超连接残差流扩展倍数
            sinkhorn_iters: Sinkhorn 迭代次数
            eps: 数值稳定性的小值

        Returns:
            pre:  (batch_size, seq_len, hc_mult)
            post: (batch_size, seq_len, hc_mult)
            comb: (batch_size, seq_len, hc_mult, hc_mult)
        """
        batch_size, seq_len, _ = mixes.shape

        mixes = mixes.float()
        hc_scale = hc_scale.float()
        hc_base = hc_base.float()

        # split mixes
        pre_logits = mixes[..., :hc_mult]  # (batch_size, seq_len, hc_mult)
        post_logits = mixes[..., hc_mult : 2 * hc_mult]  # (batch_size, seq_len, hc_mult)
        comb_logits = mixes[..., 2 * hc_mult :]  # (batch_size, seq_len, hc_mult * hc_mult)

        # pre = sigmoid(x * scale + base) + eps
        pre_base = hc_base[:hc_mult]
        pre = torch.sigmoid(pre_logits * hc_scale[0] + pre_base) + eps  # 确保 pre 严格大于 0

        # post = 2 * sigmoid(x * scale + base)
        post_base = hc_base[hc_mult : 2 * hc_mult]
        post = 2.0 * torch.sigmoid(post_logits * hc_scale[1] + post_base)

        # comb (batch_size, seq_len, hc_mult * hc_mult) -> (batch_size, seq_len, hc_mult, hc_mult)
        comb_base = hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
        comb = comb_logits.view(batch_size, seq_len, hc_mult, hc_mult)
        comb = comb * hc_scale[2] + comb_base

        # exp(comb) 和 Sinkhorn 归一化
        # 由于直接 exp 可能导致溢出，这里使用如下技巧：
        # 1. 先对 comb 应用 softmax 即 torch.softmax(comb, dim=-1) + eps，这一步的 softmax 通常是数值稳定的
        # 它等价于： 
        #   comb = torch.exp(comb)
        #   comb = comb / comb.sum(dim=-1, keepdim=True)
        #   comb = comb + eps  (确保为正)
        # 因此，它相当于通过 softmax 实现了 exp(comb)，同时顺便进行了一次行归一化
        # 2. 然后紧接着做一次列归一化：comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        # 之后开始剩下的 sinkhorn_iters - 1 次 Sinkhorn 迭代:
        # repeat sinkhorn_iters - 1 times:
        #   normalize rows
        #   normalize columns
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        return pre, post, comb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 输入的 hidden_states 形状为已经经过扩展的残差流 (batch_size, seq_len, hc_mult, hidden_size)
        # hc_pre 将 hidden_states 从残差流的形状转换为 (batch_size, seq_len, hidden_size)
        # 经过 attention 或 ffn 后，再通过 hc_post 将其转换回残差流的形状 (batch_size, seq_len, hc_mult, hidden_size)
        
        # attention
        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        
        hidden_states = self.attn_norm(hidden_states)  # (batch_size, seq_len, hidden_size)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            past_key_values=past_key_values if use_cache else None,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        # feedforward
        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states, seq_aux_loss, global_counts = self.ffn(hidden_states, input_ids=input_ids)
        
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        return hidden_states, seq_aux_loss, global_counts


# mini_deepseekv4 抽象基类
class MiniDeepSeekV4PreTrainedModel(PreTrainedModel):
    config: MiniDeepSeekV4Config  # 用于类型标注(type hint)
    base_model_prefix = "model"  # 定义模型主干模块的属性名
    config_class = MiniDeepSeekV4Config  # 用于 transformers 框架的模型注册机制，类属性(class level)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)  # 调用 PreTrainedModel 类的初始化方法
        std = self.config.initializer_range

        if isinstance(module, MiniDeepSeekV4Gate):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        for attr in ("hc_attn_fn", "hc_ffn_fn", "hc_head_fn"):
            if hasattr(module, attr):
                nn.init.normal_(getattr(module, attr), mean=0.0, std=std)
        for attr in ("hc_attn_base", "hc_ffn_base", "hc_head_base"):
            if hasattr(module, attr):
                nn.init.zeros_(getattr(module, attr))
        for attr in ("hc_attn_scale", "hc_ffn_scale", "hc_head_scale"):
            if hasattr(module, attr):
                nn.init.ones_(getattr(module, attr))
        
        if hasattr(module, "attn_sink"):
            nn.init.zeros_(module.attn_sink)
            
        if hasattr(module, "position_bias"):
            nn.init.zeros_(module.position_bias)


class MiniDeepSeekV4Model(MiniDeepSeekV4PreTrainedModel):
    def __init__(self, config: MiniDeepSeekV4Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.hc_mult = config.hc_mult

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniDeepSeekV4DecoderLayer(layer_idx, config) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))

        self.post_init()

    def hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        dtype = hidden_states.dtype
        
        flat_states = hidden_states.flatten(2)  # (batch_size, seq_len, hc_mult * hidden_size)
        rsqrt = torch.rsqrt(flat_states.float().square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps)  # RMSNorm
        mixes = F.linear(flat_states, self.hc_head_fn).float() * rsqrt  # (batch_size, seq_len, hc_mult)
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + self.config.hc_eps  # 确保 pre 严格大于 0
        
        output = torch.sum(pre.to(dtype).unsqueeze(-1) * flat_states.view(shape), dim=2)  # (batch_size, seq_len, hidden_size)
        return output.to(dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> MiniDeepSeekV4ModelOutput:
        # ^ 是异或运算符，只有一个是 True 时为 True，即 input_ids 和 inputs_embeds 只能提供一个
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 获取嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 如果使用缓存且 past_key_values 为空，则初始化自定义 Cache，并为每一层创建一个 MiniDeepSeekV4CacheLayer
        if use_cache and past_key_values is None:
            past_key_values = Cache(layers=[MiniDeepSeekV4CacheLayer(self.config) for _ in range(self.config.num_hidden_layers)])

        # cache_position 是当前输入序列的位置索引，索引范围为 [past_seen_tokens, past_seen_tokens + seq_len]
        # 形状为 (seq_len,)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(0) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # position_ids 同样是位置索引，形状为 (batch_size, seq_len)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        # 原始的 padding mask 是一个二维张量，形状为 (batch_size, seq_len)，其中 1 表示有效 token，0 表示 padding token
        padding_mask = attention_mask if attention_mask is not None and attention_mask.ndim == 2 else None
        causal_mask = create_sliding_window_causal_mask(
            self.config,
            inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # 扩展残差流 (batch_size, seq_len, hc_mult, hidden_size)
        total_seq_aux_loss = None  # 用于记录所有 MoE 层的序列级别辅助损失
        all_global_counts = []  # 用于记录每个 MoE 层的专家激活次数

        for decoder_layer in self.layers:
            hidden_states, seq_aux_loss, global_counts = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
                input_ids=input_ids,
                **kwargs,
            )
            if seq_aux_loss is not None:
                total_seq_aux_loss = seq_aux_loss if total_seq_aux_loss is None else total_seq_aux_loss + seq_aux_loss
            if global_counts is not None:
                all_global_counts.append({"layer_idx": decoder_layer.layer_idx, "global_counts": global_counts.detach().cpu().tolist()})

        hidden_states_for_mtp = hidden_states  # 用于 MTP 模块的输入
        hidden_states = self.norm(self.hc_head(hidden_states))  # 聚合残差流并归一化，得到最终的 hidden states (batch_size, seq_len, hidden_size)

        return MiniDeepSeekV4ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states_for_mtp=hidden_states_for_mtp,
            total_seq_aux_loss=total_seq_aux_loss,
            all_global_counts=all_global_counts,
        )


class MiniDeepSeekV4MTP(nn.Module):
    """
    多 token 预测 (Multi-Token Prediction, MTP)
    """
    def __init__(self, config: MiniDeepSeekV4Config, embed_tokens: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.transformer_block = MiniDeepSeekV4DecoderLayer(0, config)  # 使用 hash routing 的 MoE

        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))

    def hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        dtype = hidden_states.dtype
        
        flat_states = hidden_states.flatten(2)
        rsqrt = torch.rsqrt(flat_states.float().square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = F.linear(flat_states, self.hc_head_fn).float() * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + self.config.hc_eps
        
        output = torch.sum(pre.to(dtype).unsqueeze(-1) * flat_states.view(shape), dim=2)
        return output.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, mtp_seq_len = input_ids.shape
        
        inputs_embeds = self.input_norm(self.embed_tokens(input_ids))  # (batch_size, mtp_seq_len, hidden_size)
        last_hidden_states = self.hidden_norm(last_hidden_states)  # (batch_size, mtp_seq_len, hc_mult, hidden_size)
        hidden_states = self.e_proj(inputs_embeds).unsqueeze(2) + self.h_proj(last_hidden_states)  # (batch_size, mtp_seq_len, hc_mult, hidden_size)

        # 由于只在预训练期间使用 mtp 模块，position_ids 直接从 0 开始递增即可，形状为 (batch_size, mtp_seq_len)
        if position_ids is None:
            position_ids = torch.arange(mtp_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, mtp_seq_len)
        cache_position = position_ids[0]
        
        # 由于只在预训练期间使用 mtp 模块，attention_mask 直接设置为全 1 即可，形状为 (batch_size, mtp_seq_len)
        attention_mask = torch.ones(batch_size, mtp_seq_len, device=input_ids.device)
        padding_mask = attention_mask
        causal_mask = create_sliding_window_causal_mask(
            self.config,
            inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states, _, _ = self.transformer_block(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            padding_mask=padding_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=cache_position,
            use_cache=False,
            input_ids=input_ids,
        )
        logits = self.lm_head(self.output_norm(self.hc_head(hidden_states)))
        return logits, hidden_states


class MiniDeepSeekV4ForCausalLM(MiniDeepSeekV4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 声明需要共享的权重
    architecture_type = "MoE"  # 自定义字段

    def __init__(self, config: MiniDeepSeekV4Config):
        super().__init__(config)
        self.model = MiniDeepSeekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = MiniDeepSeekV4MTP(config, self.model.embed_tokens, self.lm_head) if config.use_mtp else None

        # 如果使用 MTP，需要声明共享的权重
        if self.mtp is not None:
            self._dynamic_tied_weights_keys = [
                "model.embed_tokens.weight",
                "mtp.embed_tokens.weight",
                "lm_head.weight",
                "mtp.lm_head.weight",
            ]

        self.post_init()

    def remove_mtp_module(self):
        """
        MTP 模块用于辅助模型在预训练时, 具备一定预测未来几个 token 的能力, 预训练结束后, 删除 MTP 模块, 保存不含 MTP 模块的模型
        在后续微调及推理时, 默认 use_mtp=False, 模型结构在初始化时已经不包含 MTP 模块
        
        此方法会：
         1. 删除 MTP 模块
         2. 更新配置为 use_mtp=False
         3. 清理共享权重声明
        """
        if self.mtp is not None:
            del self.mtp
            self.mtp = None
            self.config.use_mtp = False
            if hasattr(self, "_dynamic_tied_weights_keys"):
                del self._dynamic_tied_weights_keys

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> MiniDeepSeekV4ForCausalLMOutput:
        # ----------------- 主模型部分 -----------------
        # 主干模型前向传播
        outputs: MiniDeepSeekV4ModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

        # 主模型输出 logits
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        main_loss = None  # 训练阶段
        if labels is not None:
            # transformers 的 loss_function 会在内部对 label 进行 shift 操作
            # 需注意这里的 labels 是还未进行 shift 的，实际上就是 input_ids 本身
            # 详见 transformer.loss.loss_utils.py 的 ForCausalLMLoss
            main_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # ----------------- MTP 模块部分 -----------------
        # 此处仅使用固定预测深度为 1 的逻辑，即只使用 1 个 MTP 模块
        # MTP 使用 seq[:-2] 做下两个 token 的预测，预测目标为 seq[2:]
        mtp_loss = None
        if self.config.use_mtp and self.mtp is not None and self.training and input_ids is not None and input_ids.shape[1] > 1:
            # 这里的数据构造逻辑兼容 self.loss_function 的 shift 操作
            # 例如，主模型的 input_ids 为 [1, 2, 3, 4, 5]，它的 shift 后的 labels 为 [2, 3, 4, 5, x]，其中 x 是 ignore_index
            # 那么，mtp 的辅助输入为 [2, 3, 4, 5]，预测目标为 [3, 4, 5, x]，主模型使用 [1, 2, 3, 4] 所对应的输出 hidden states
            # 因此，在 mtp 的作用下，主模型的 [1, 2, 3, 4] 的预测目标为 [3, 4, 5, x]，4 没有实际的预测目标，因此被 ignore_index 覆盖
            mtp_input_ids = input_ids[:, 1:]  # mtp 的辅助输入，形状为 (batch_size, seq_len - 1)，它执行的是 next token prediction
            mtp_label = input_ids[:, 1:]  # mtp 的预测目标，由于 self.loss_function 会在内部对 label 进行 shift 操作，因此这里实际就是 mtp_input_ids 本身
            mtp_logits, _ = self.mtp(
                input_ids=mtp_input_ids,
                last_hidden_states=outputs.hidden_states_for_mtp[:, :-1],  # 主模型的输出 hidden states，形状为 (batch_size, seq_len - 1, hidden_size)
            )
            mtp_loss = self.loss_function(
                logits=mtp_logits,
                labels=mtp_label,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        # ---------------- 计算总损失 ----------------
        loss = main_loss
        if mtp_loss is not None:
            loss = mtp_loss * self.config.mtp_loss_lambda if loss is None else loss + mtp_loss * self.config.mtp_loss_lambda
        if outputs.total_seq_aux_loss is not None:
            loss = outputs.total_seq_aux_loss if loss is None else loss + outputs.total_seq_aux_loss

        return MiniDeepSeekV4ForCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            total_seq_aux_loss=outputs.total_seq_aux_loss,
            total_mtp_loss=mtp_loss * self.config.mtp_loss_lambda if mtp_loss is not None else None,
            all_global_counts=outputs.all_global_counts,
        )
