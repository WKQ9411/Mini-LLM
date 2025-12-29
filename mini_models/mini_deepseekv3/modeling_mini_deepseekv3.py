import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from dataclasses import dataclass
from typing import Union, Optional, Tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin

from ..base_module import RMSNorm, SwiGLUFFN
from ..attention import MultiHeadLatentAttention
from ..rope import RotaryEmbedding
from .configuration_mini_deepseekv3 import MiniDeepSeekV3Config

# 参考代码:
# - deepseek 官方仓库源码: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
# - transformers 源码: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py


# mini_deepseekv3 专家网络
class MiniDeepSeekV3Expert(SwiGLUFFN):
    """
    mini_deepseekv3 专家网络, 结构实际上就是 SwiGLUFFN
    """
    pass


# mini_deepseekv3 前馈网络
class MiniDeepSeekV3MLP(SwiGLUFFN):
    """
    mini_deepseekv3 前馈网络, 结构实际上就是 SwiGLUFFN
    """
    pass


# mini_deepseekv3 自定义输出类
@dataclass
class MiniDeepSeekV3ModelOutput(BaseModelOutputWithPast):
    """自定义输出类，包含额外的字段"""
    hidden_states_for_mtp: Optional[torch.FloatTensor] = None
    total_seq_aux_loss: Optional[torch.Tensor] = None
    all_global_counts: Optional[list[dict]] = None


# mini_deepseekv3 自定义输出类
@dataclass
class MiniDeepSeekV3ForCausalLMOutput(CausalLMOutputWithPast):
    """自定义输出类，包含额外的字段"""
    total_seq_aux_loss: Optional[torch.Tensor] = None
    total_mtp_loss: Optional[torch.Tensor] = None
    all_global_counts: Optional[list[dict]] = None


# mini_deepseekv3 门控网络
class MiniDeepSeekV3Gate(nn.Module):
    """
    即 Router, MoE 中的门控网络, 用于动态路由, 整体过程是: 
     1. 对专家进行分组, 共 n_groups 个组
     2. 每个组计算 2 个最大亲和度得分之和
     3. 根据上述结果选出 topk_groups 个组
     4. 从上述 topk_groups 个组的所有专家中, 选出 topk 个专家

    Attributes:
        topk (int): 每个输入激活的专家数量
        n_groups (int): 专家的分组数量
        topk_groups (int): 选中 topk 个分组
        route_scale (float): 路由权重的缩放因子
        n_routed_experts (int): 路由的专家数量
        weight (torch.nn.Parameter): 门控的可学习权重参数
        bias (torch.nn.Parameter): 门控的偏置项
        use_noaux_load_balance (bool): 是否使用无辅助损失的负载均衡策略
        original_scores (torch.Tensor): 原始的亲和度得分, 形状为 (batch_size * seq_len, n_routed_experts)
    """
    def __init__(self, config: MiniDeepSeekV3Config):
        """
        门控网络初始化

        Args:
            config (MiniDeepSeekV3Config): 模型配置参数
        """
        super().__init__()
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.route_scale = config.route_scale
        self.n_routed_experts = config.n_routed_experts
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))  # torch.empty 用于创建未初始化的张量，需要手动初始化
        self.bias = nn.Parameter(torch.empty(config.n_routed_experts), requires_grad=False)  # 用于无辅助损失负载均衡策略的 bias，不参与梯度计算，基于策略来更新bias，可以理解为通过策略干预而不是 loss 来进行更新的模型参数
        self.use_noaux_load_balance = config.use_noaux_load_balance
        self.original_scores = None  # 用于存储原始的亲和度得分, 形状为 (batch_size * seq_len, n_routed_experts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化参数
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.use_noaux_load_balance:
            nn.init.zeros_(self.bias)  # 这里将 bias 初始化为 0

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        门控网络的前向传播

        Args:
            hidden_states (torch.Tensor): 输入按 token 排列, 形状为 (batch_size * seq_len, hidden_size), 在输入前已经调整好形状

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 路由权重和选择的专家索引, 形状均为 (batch_size * seq_len, topk)
        """
        scores_logits = F.linear(hidden_states, self.weight, None)  # 计算所有 token 对专家的亲和度得分 (batch_size * seq_len, n_routed_experts)
        scores = scores_logits.sigmoid()  # 使用 sigmoid 来计算亲和度得分 (batch_size * seq_len, n_routed_experts)
        self.original_scores = scores  # 保留原始得分，用于后续根据原始得分抽取 topk 个专家 (batch_size * seq_len, n_routed_experts)
        scores_for_topk = scores.clone() # 创建一个新变量专门用于专家选择，避免影响梯度回传
        
        if self.use_noaux_load_balance:
            scores_for_topk = scores_for_topk + self.bias
        
        # 为专家分组，并选出 topk_groups 个组
        if self.n_groups > 1:
            scores_view = scores_for_topk.view(hidden_states.size(0), self.n_groups, -1)  # (batch_size * seq_len, n_groups, n_routed_experts_per_group)
            if not self.use_noaux_load_balance:
                group_scores = scores_view.amax(dim=-1)  # 取每个组的得分最大值，形状变为 (batch_size * seq_len, n_groups)
            else:
                group_scores = scores_view.topk(2, dim=-1)[0].sum(dim=-1)  # 取每个组最大两个得分的和，形状变为 (batch_size * seq_len, n_groups)
            indices_groups = group_scores.topk(self.topk_groups, dim=-1)[1]  # 获取组得分在前 topk_groups 的索引 (batch_size * seq_len, topk_groups)
            mask = torch.ones(hidden_states.size(0), self.n_groups, dtype=torch.bool, device=hidden_states.device)
            mask.scatter_(dim=1, index=indices_groups, value=False)  # 生成 mask 用于标记哪些组被选中，这里将选中的组设置为 False (batch_size * seq_len, n_groups)
            # 首先将 mask 增加最后一个维度，变为(batch_size * seq_len, n_groups, 1)，以适应分数张量(batch_size * seq_len, n_groups, n_routed_experts_per_group)
            # 将对应 mask 为 True 的，也就是未选中的组的所有专家得分置为负无穷，这样就只保留了选中组的所有专家的得分，并展平为(batch_size * seq_len, n_routed_experts)
            scores_for_topk = scores_view.masked_fill(mask=mask.unsqueeze(-1), value=float("-inf")).flatten(1)
        
        # 选出 topk 个专家，得到其索引和权重
        _, indices = torch.topk(scores_for_topk, self.topk, dim=-1)  # 在所选到的 topk_groups 个组的所有专家中，选出 topk 个专家的索引 (batch_size * seq_len, topk)
        weights = scores.gather(dim=1, index=indices)  # 从原始分数中按选出的索引抽取出亲和度得分，即权重 (batch_size * seq_len, topk)
        
        weights_sum = weights.sum(dim=-1, keepdim=True)
        weights = weights / (weights_sum + 1e-6)  # 将权重归一化【注意，这是在选出的 topk 中进行归一化】
        weights = weights * self.route_scale  # 应用缩放因子
            
        return weights.type_as(hidden_states), indices


# mini_deepseekv3 MoE 层
class MiniDeepSeekV3MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) 混合专家模块

    Attributes:
        hidden_size (int): 嵌入维度
        n_routed_experts (int): 路由专家数量
        n_activated_experts (int): 每个输入激活的专家数
        gate (nn.Module): 门控机制
        experts (nn.ModuleList): 专家列表
        shared_experts (nn.Module): 共享专家
        use_seq_aux (bool): 是否使用序列级别的辅助损失
        seq_aux_alpha (float): 序列级别的辅助损失的权重
        bias_update_speed (float): 偏置更新速度
    """
    def __init__(self, config: MiniDeepSeekV3Config):
        """
        MoE 初始化

        Args:
            config (MiniDeepSeekV3Config): 模型配置参数
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.gate = MiniDeepSeekV3Gate(config)
        self.experts = nn.ModuleList([MiniDeepSeekV3Expert(config.hidden_size, config.moe_intermediate_size) for _ in range(self.n_routed_experts)])
        self.shared_experts = MiniDeepSeekV3MLP(config.hidden_size, config.n_shared_experts * config.moe_intermediate_size)
        self.use_seq_aux = config.use_seq_aux
        self.seq_aux_alpha = config.seq_aux_alpha
        self.bias_update_speed = config.bias_update_speed  # 用于无辅助损失负载均衡策略的 bias 的更新速度

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        MoE 前向传播

        Args:
            hidden_states (torch.Tensor): 输入张量 (batch_size, seq_len, hidden_size)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: 输出张量 (batch_size, seq_len, hidden_size), 本层序列级辅助损失, 本层全局负载情况
        """
        shape = hidden_states.size()
        batch_size, seq_length = shape[:2]
        hidden_states = hidden_states.view(-1, self.hidden_size)  # 重新划分形状为 (batch_size * seq_len, hidden_size)
        weights, indices = self.gate(hidden_states)  # 计算得到每个 token 的路由权重和选择的专家索引，形状均为 (batch_size * seq_len, topk)
        routed_experts_output = torch.zeros_like(hidden_states)  # 用于累加路由专家的输出，形状为 (batch_size * seq_len, hidden_size)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)  # bincount 用于计算非负整数张量中每个值的出现次数，即此列表保存了一个 batch 里每个专家对应的激活次数
        global_counts = None  # 初始化全局 counts

        # -------------------- 无辅助损失负载均衡策略 --------------------
        # 这里我们计算出了一个 batch 中所有专家的激活情况，故顺便在此应用无辅助损失的负载均衡策略来更新 gate 中的 bias
        # 每一个 MoE 层更新自己的 Gate 的 bias，下一个 batch 的数据将使用更新的 bias，训练的最后一组数据更新完 bias 后，将作为模型参数保存下来
        if self.gate.use_noaux_load_balance and self.training:
            global_counts = counts.clone()
            
            # 同步所有 GPU 的 counts
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)  # 全局求和
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
        if self.use_seq_aux and self.training:
            # 计算 P_i，含义为第 i 个专家在每个 token 上的平均归一化亲和度得分
            scores_for_seq_aux = self.gate.original_scores.view(batch_size, seq_length, -1)  # 此即原始的 s_{i,t} (batch_size, seq_len, n_routed_experts)
            scores_for_seq_aux = scores_for_seq_aux / scores_for_seq_aux.sum(dim=-1, keepdim=True)  # 沿着 n_routed_experts 的方向归一化，形成 s_{i,t}'
            P_i = scores_for_seq_aux.mean(dim=1)  # 沿着 token 的方向求平均 (batch_size, n_routed_experts)

            # 计算 f_i，含义为第 i 个专家在每个 token 上的平均激活次数
            # indices 计算了一个 batch 中每个 token 激活了哪些专家，现在要计算每个序列中，每个专家被哪些 token 激活
            # 可以使用 one-hot 编码来快速计算每个专家被多少个 token 激活
            f_i = F.one_hot(indices.view(batch_size, -1), num_classes=self.n_routed_experts)  # (batch_size, seq_len * topk, n_routed_experts)
            f_i = f_i.sum(dim=1)  # 沿 seq_len * topk 维度相加后，求出每个专家被多少个 token 激活 (bsz, n_routed_experts)
            f_i = (f_i * self.n_routed_experts) / (self.n_activated_experts * seq_length)  # 计算每个专家的平均激活次数并乘以系数 (batch_size, n_routed_experts)

            seq_aux_loss = (f_i * P_i).sum() * self.seq_aux_alpha  # 计算序列级别的辅助损失
        else:
            seq_aux_loss = None  # 不使用序列级别的辅助损失时，将损失设为 None

        # 为每个 token 计算路由专家的输出和
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)  # 找到激活了第 i 个专家的 token，idx 代表行索引(即第几个 token)，top 代表列索引(即该 token 的 top 几选择)，idx 和 top 的类型为 torch.Tensor
            # 假设 n_matches 是匹配当前专家的 token 数量，那么 expert(x[idx]) 的形状是 (n_matches, hidden_size)，weights[idx, top] 的形状是 (n_matches,)
            # None 用于增加一个维度，使形状变为 (n_matches, 1)
            routed_experts_output[idx] += expert(hidden_states[idx]) * weights[idx, top, None]  # 如果 idx 是标量(如 idx=5 )，结果形状是 (dim,)；如果 idx 是单元素张量(如 idx=tensor([5]) )，结果形状是 (1, dim)
        
        # 计算共享专家的输出
        shared_experts_output = self.shared_experts(hidden_states)  # 形状为 (batch_size * seq_len, hidden_size)
 
        return (routed_experts_output + shared_experts_output).view(shape), seq_aux_loss, global_counts


# mini_deepseekv3 解码器层
class MiniDeepSeekV3DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, config: MiniDeepSeekV3Config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx

        self.self_attn = MultiHeadLatentAttention(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            attention_bias=config.attention_bias,
            attn_impl=config.attn_impl,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = MiniDeepSeekV3MLP(config.hidden_size, config.intermediate_size) if layer_idx < config.num_dense_layers else MiniDeepSeekV3MoE(config)  # 前几层是常规的 MLP 层
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # feedforward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, MiniDeepSeekV3MoE):
            hidden_states, seq_aux_loss, global_counts = self.mlp(hidden_states)  # 如果是 MoE 层，返回 hidden_states, seq_aux_loss 和 global_counts
            hidden_states = residual + hidden_states
            return hidden_states, seq_aux_loss, global_counts
        else:
            hidden_states = self.mlp(hidden_states)  # 如果是非 MoE 层，seq_aux_loss 和 global_counts 为 None
            hidden_states = residual + hidden_states
            return hidden_states, None, None


# mini_deepseekv3 抽象基类
class MiniDeepSeekV3PreTrainedModel(PreTrainedModel):
    config: MiniDeepSeekV3Config  # 用于类型标注(type hint)
    base_model_prefix = "model"  # 定义模型主干模块的属性名
    config_class = MiniDeepSeekV3Config  # 用于 transformers 框架的模型注册机制，类属性(class level)


# mini_deepseekv3 主干模型
class MiniDeepSeekV3Model(MiniDeepSeekV3PreTrainedModel):
    def __init__(self, config: MiniDeepSeekV3Config):
        super().__init__(config)  # 调用父类初始化方法，会有 self.config = config，实例属性(instance level)
        self.vocab_size = config.vocab_size
        self.padding_idx = getattr(config, "pad_token_id", None)

        # 词嵌入层
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,  # pad token 不应参与嵌入计算
        )

        # 多层解码器
        self.layers = nn.ModuleList(
            [
                MiniDeepSeekV3DecoderLayer(layer_idx, config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # 输出归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 旋转位置编码层
        self.rotary_emb = RotaryEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.qk_rope_head_dim,
            rope_theta=config.rope_theta,
        )

        # 调用父类方法，其中主要会进行：
        #   - init_weights 初始化权重
        #   - tie_weights 将输入 Embedding 和 输出 lm_head 进行权重共享，使其语义空间一致，此外还能减小参数量
        self.post_init()
    
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
    ) -> BaseModelOutputWithPast:
        """
        前向传播

        Args:
            input_ids: 输入 token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, 1, q_len, kv_len) 或 (batch_size, number_of_seen_tokens + q_len)
            position_ids: 位置索引 (batch_size, seq_len)
            past_key_values: 继承自 Cache 基类, 用于缓存和管理 KV Cache
            inputs_embeds: 嵌入向量 (batch_size, seq_len, hidden_size)
            cache_position: 缓存位置索引 (seq_len,) 或 (batch_size, seq_len)
            use_cache: 是否使用缓存

        Returns:
            BaseModelOutputWithPast: 包含 hidden_states, past_key_values 和其他自定义的输出
        """
        # ^ 是异或运算符，只有一个是 True 时为 True，即 input_ids 和 inputs_embeds 只能提供一个
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 获取嵌入向量
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        # DynamicCache 用于动态缓存 KV 值，StaticCache 则是静态预留
        # TODO: 当前直接使用 Cache 类，后续可以进行一些兼容的自定义实现
        # 参考 DynamicCache 的注释:
        # A cache that grows dynamically as more tokens are generated. This is the default for generative models.
        # It stores the key and value states as a list of `CacheLayer`, one for each layer.
        # The expected shape for each tensor in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # cache_position 是当前输入序列的位置索引，索引范围为 [past_seen_tokens, past_seen_tokens + seq_len]
        # 形状为 (seq_len,)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # position_ids 同样是位置索引，形状为 (batch_size, seq_len)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        # 创建因果掩码
        # TODO: 当前直接使用 transformers 的实现，后续可以进行一些兼容的自定义实现
        # attention_mask 的含义参考 create_causal_mask 的注释:
        # The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens + q_length).
        # It can also be an already prepared 4D mask, in which case it is returned as-is.
        # 即默认情况下是一个 2D 的 pad mask，形状为 (batch_size, number_of_seen_tokens + q_len)，通常就是 (batch_size, kv_len)
        # 如果已经是一个准备好的 4D mask，则直接原样返回，形状是 (batch_size, 1, q_len, kv_len)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # 解码器逐层前向传播
        total_seq_aux_loss = None  # 用于记录所有 MoE 层的序列级别辅助损失
        all_global_counts = []  # 用于记录每个 MoE 层的专家激活次数
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, seq_aux_loss, global_counts = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            if seq_aux_loss is not None:
                if total_seq_aux_loss is None:
                    total_seq_aux_loss = seq_aux_loss
                else:
                    total_seq_aux_loss += seq_aux_loss
            if global_counts is not None:
                all_global_counts.append({"layer_idx": decoder_layer.layer_idx, "global_counts": global_counts.cpu().tolist()})

        # 输出归一化
        hidden_states_for_mtp = hidden_states  # 用于 MTP 模块的输入
        hidden_states = self.norm(hidden_states)

        return MiniDeepSeekV3ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            # 自定义字段
            hidden_states_for_mtp=hidden_states_for_mtp,
            total_seq_aux_loss=total_seq_aux_loss,
            all_global_counts=all_global_counts,
        )


# mini_deepseekv3 MTP 层
class MiniDeepSeekV3MTP(nn.Module):
    """
    多 token 预测 (Multi-Token Prediction, MTP)

    Attributes:
        config (MiniDeepSeekV3Config): 模型配置参数
        embed_tokens (nn.Module): 嵌入层
        lm_head (nn.Module): 输出投影
        hidden_norm (nn.Module): 对上一个 MTP 模块或主模型输出的 hidden state 应用的 Layer Normalization
        input_norm (nn.Module): 对本 MTP 模块输入应用的 Layer Normalization
        output_norm (nn.Module): 对本 MTP 模块输出应用的 Layer Normalization
        linear_proj (nn.Module): 线性投影层，用于将 MTP 模块的两个输入拼接后进行投影
        transformer_block (MiniDeepSeekV3DecoderLayer): Transformer Block
    """
    def __init__(self, config: MiniDeepSeekV3Config, embed_tokens: nn.Module, lm_head: nn.Module):
        """
        初始化 MTP

        Args:
            config (MiniDeepSeekV3Config): 模型配置参数
            embed_tokens (nn.Module): 嵌入层, 来自于 MiniDeepSeekV3ForCausalLM 共享
            lm_head (nn.Module): 输出投影, 来自于 MiniDeepSeekV3ForCausalLM 共享
        """
        super().__init__()
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.hidden_norm = RMSNorm(config.hidden_size)
        self.input_norm = RMSNorm(config.hidden_size)
        self.output_norm = RMSNorm(config.hidden_size)  # 原文的图中在每个 MTP 的 Transformer Block 后没有进行 norm，但主模型有，这里也加上
        self.linear_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.transformer_block = MiniDeepSeekV3DecoderLayer(0, config)  # 原文中未说明 MTP 使用的是否是 MoE 架构，这里我们通过硬编码 layer_idx=0 的方式使用普通 MLP 作为前馈
        # mtp 使用的旋转位置编码层
        self.rotary_emb = RotaryEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.qk_rope_head_dim,
            rope_theta=config.rope_theta,
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        last_hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MTP 的前向传播

        Args:
            input_ids (torch.Tensor): 输入进 MTP 模块的 token ids, 形状为 (batch_size, mtp_seq_len)
            last_hidden_states (torch.Tensor): 上一个 MTP 模块或主模型的 last hidden states 输出 (batch_size, mtp_seq_len, hidden_size)
            position_ids (Optional[torch.LongTensor]): 位置索引 (batch_size, mtp_seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输出 Logits (batch_size, vocab_size) 和给下一个 MTP 使用的 h (batch_size, mtp_seq_len, dim)
        """
        batch_size, mtp_seq_len = input_ids.shape

        inputs_embeds = self.embed_tokens(input_ids)  # (batch_size, mtp_seq_len, hidden_size)
        input_hidden_states = self.input_norm(inputs_embeds)
        last_hidden_states = self.hidden_norm(last_hidden_states)
        hidden_states = self.linear_proj(torch.cat([input_hidden_states, last_hidden_states], dim=-1))  # 拼接后进行投影 (batch_size, mtp_seq_len, hidden_size)

        # 直接创建因果 mask，当前只在训练期间使用 mtp 模块，形状为 (batch_size, 1, mtp_seq_len, mtp_seq_len)
        causal_mask = None
        if mtp_seq_len > 1:
            causal_mask = torch.zeros((batch_size, 1, mtp_seq_len, mtp_seq_len), device=input_ids.device, dtype=torch.float32)
            causal_mask.masked_fill_(
                torch.triu(torch.ones((mtp_seq_len, mtp_seq_len), device=input_ids.device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0),
                torch.finfo(torch.float32).min
            )
        
        # 由于只在训练期间使用 mtp 模块，position_ids 直接从 0 开始递增即可，形状为 (batch_size, mtp_seq_len)
        if position_ids is None:
            position_ids = torch.arange(mtp_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, mtp_seq_len)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)  # cos, sin 表, 形状为 (batch_size, mtp_seq_len, head_dim)
        
        hidden_states, _, _ = self.transformer_block(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            )  # (batch_size, mtp_seq_len, hidden_size)

        last_hidden_states = hidden_states  # 输出 hidden_states 作为下一个 MTP 模块的输入
        hidden_states = self.output_norm(hidden_states)  # (batch_size, mtp_seq_len, hidden_size)
        mtp_logits = self.lm_head(hidden_states)  # (batch_size, mtp_seq_len, vocab_size)

        return mtp_logits, last_hidden_states


# mini_deepseekv3 因果语言模型
class MiniDeepSeekV3ForCausalLM(MiniDeepSeekV3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]  # 声明需要共享的权重
    architecture_type = "MoE"  # 自定义字段

    def __init__(self, config: MiniDeepSeekV3Config):
        super().__init__(config)
        self.model = MiniDeepSeekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = MiniDeepSeekV3MTP(config, self.model.embed_tokens, self.lm_head) if config.use_mtp else None  # 共享词嵌入层和输出投影层

        # 如果使用 MTP，需要声明共享的权重
        if self.mtp is not None:
            if not hasattr(self, '_dynamic_tied_weights_keys'):
                self._dynamic_tied_weights_keys = []
            # 添加共享关系
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
            if hasattr(self, '_dynamic_tied_weights_keys'):
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
    ) -> CausalLMOutputWithPast:
        """
        前向传播

        Args:
            input_ids: 输入 token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, 1, q_len, kv_len) 或 (batch_size, number_of_seen_tokens + q_len)
            position_ids: 位置索引 (batch_size, seq_len)
            past_key_values: 继承自 Cache 基类, 用于缓存和管理 KV Cache
            inputs_embeds: 输入嵌入向量 (batch_size, seq_len, hidden_size)
            labels: 用于计算损失的目标 token ids (batch_size, seq_len)
            use_cache: 是否使用缓存
            cache_position: 缓存位置索引 (seq_len,) 或 (batch_size, seq_len)
            logits_to_keep: 保留的 logits 数量，默认为 0 表示不进行任何过滤

        Returns:
            CausalLMOutputWithPast: 包含 logits 和损失的字典
        """
        # ----------------- 主模型部分 -----------------
        # 主干模型前向传播
        outputs: MiniDeepSeekV3ModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # 解包输出
        hidden_states = outputs.last_hidden_state
        hidden_states_for_mtp = outputs.hidden_states_for_mtp
        total_seq_aux_loss = outputs.total_seq_aux_loss
        all_global_counts = outputs.all_global_counts

        # 关于 logits_to_keep，transformers 中的描述如下：
        #   If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
        #   `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
        #   token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
        #   If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
        #   This is useful when using packed tensor format (single dimension for batch and sequence length).
        # 因此训练时应计算全部的 logits，即 logits_to_keep = 0，推理时一般只取最后一个 logits，即 logits_to_keep = 1
        # slice(-logits_to_keep, None) 等价于 -logits_to_keep:
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        main_loss = None
        if labels is not None:  # 训练阶段
            # transformers 的 loss_function 会在内部对 label 进行 shift 操作
            # 需注意这里的 labels 是还未进行 shift 的，实际上就是 input_ids 本身
            # 详见 transformer.loss.loss_utils.py 的 ForCausalLMLoss
            main_loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        # ----------------- MTP 模块部分 -----------------
        # 此处仅使用固定预测深度为 1 的逻辑，即只使用 1 个 MTP 模块
        # MTP 使用 seq[:-2] 做下两个 token 的预测，预测目标为 seq[2:]
        mtp_loss = None
        if self.mtp is not None and self.training:
            # 这里的数据构造逻辑兼容 self.loss_function 的 shift 操作
            # 例如，主模型的 input_ids 为 [1, 2, 3, 4, 5]，它的 shift 后的 labels 为 [2, 3, 4, 5, x]，其中 x 是 ignore_index
            # 那么，mtp 的辅助输入为 [2, 3, 4, 5]，预测目标为 [3, 4, 5, x]，使用的主模型 hidden states 为 [1, 2, 3, 4] 所对应的输出 hidden states
            # 因此，在 mtp 的作用下，主模型的 [1, 2, 3, 4] 的预测目标为 [3, 4, 5, x]，4 没有实际的预测目标，因此被 ignore_index 覆盖
            mtp_input_ids = input_ids[:, 1:]  # mtp 的辅助输入，形状为 (batch_size, seq_len - 1)，它执行的是 next token prediction
            mtp_label = input_ids[:, 1:]  # mtp 的预测目标，由于 self.loss_function 会在内部对 label 进行 shift 操作，因此这里实际就是 mtp_input_ids 本身
            mtp_logits, _ = self.mtp(
                input_ids=mtp_input_ids,
                last_hidden_states=hidden_states_for_mtp[:, :-1, :],  # 主模型的输出 hidden states，形状为 (batch_size, seq_len - 1, hidden_size)
            )

            mtp_loss = self.loss_function(
                logits=mtp_logits,
                labels=mtp_label,  # 在 self.loss_function 中会自动进行 shift 操作
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        
        # ---------------- 计算总损失 ----------------
        if mtp_loss and total_seq_aux_loss:
            loss = main_loss + mtp_loss * self.config.mtp_loss_lambda + total_seq_aux_loss
        elif mtp_loss and not total_seq_aux_loss:
            loss = main_loss + mtp_loss * self.config.mtp_loss_lambda
        elif not mtp_loss and total_seq_aux_loss:
            loss = main_loss + total_seq_aux_loss
        else:
            loss = main_loss
        
        return MiniDeepSeekV3ForCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # 自定义字段
            total_seq_aux_loss=total_seq_aux_loss,
            total_mtp_loss=mtp_loss * self.config.mtp_loss_lambda if mtp_loss is not None else None,
            all_global_counts=all_global_counts,
        )