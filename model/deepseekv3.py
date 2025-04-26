import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from .basemodel import BaseModelArgs, BaseModel

# 【说明】
# 1. 主要从MLA、DeepSeekMoE、MTP来实现一个mini-deepseek-v3架构模型
# 2. 由于模型较小，不考虑通过YaRN来扩展上下文长度，使用原始RoPE
# 3. 不考虑原文中负载均衡策略、FP8混合精度训练、DualPipe等工程优化
# 4. 由于模型参数量较少，采用单卡或ddp的方式训练
# 5. 根据以上几点对源代码进行一定修改与简化，无需修改的地方尽量与源码保持一致
# 6. 本项目修改自DeepSeek-V3源码：https://github.com/deepseek-ai/DeepSeek-V3 ，而非HuggingFace版本的代码


# ----------------------------------------------【参数配置】---------------------------------------------- #
@dataclass
class DeepSeekV3ModelArgs(BaseModelArgs):
    """
    配置模型参数

    Attributes:
        max_batch_size (int): 最大批量大小
        max_seq_len (int): 最大序列长度
        vocab_size (int): 词典大小
        dim (int): 嵌入维度
        inter_dim (int): MLP层的中间维度
        moe_inter_dim (int): MoE层专家的中间维度
        n_layers (int): Transformer层的数量
        n_dense_layers (int): 模型中密集层的数量, 前几层负载均衡收敛慢, 因此设置为密集层
        n_heads (int): 注意力头数
        n_routed_experts (int): MoE层中路由的专家数量
        n_shared_experts (int): MoE层中共享的专家数量
        n_activated_experts (int): MoE层中激活的专家数量
        n_expert_groups (int): 专家的分组数量
        n_limited_groups (int): 路由限制, 每次最多从限制的专家组里选择专家
        route_scale (float): 路由权重的缩放因子
        use_noaux_tc (bool): 是否使用无辅助损失的负载均衡策略
        bias_update_speed (float): 偏置更新速度
        use_seq_aux (bool): 是否使用序列级别的辅助损失
        seq_aux_alpha (float): 序列级别的辅助损失的权重
        q_lora_rank (int): query 的下投影维度【对应论文中的 d_c'】
        kv_lora_rank (int): key/value 的下投影维度【对应论文中的 d_c】
        qk_nope_head_dim (int): 没有位置编码的 q_t^C 和 k_t^C 的每个头的维度【对应论文中的 d_h】
        qk_rope_head_dim (int): 解耦的带 RoPE 的 q_t^R 和 k_t^R 的每个头的维度【对应论文中的 d_h^R】
        v_head_dim (int): value 的每个头的维度, 可以与 qk_nope_head_dim 不同【但在论文中也同样设定为 d_h】
        rope_theta (float): 旋转位置编码的基底【即 θ_d=b^(-2d/D) 中的 b】
        use_mtp (bool): 是否使用 MTP 策略
        mtp_loss_lambda (float): MTP 损失的权重
    """
    max_batch_size: int = 16
    max_seq_len: int = 512
    vocab_size: int = -1  # 加载模型时传入
    dim: int = 768
    inter_dim: int = 3072
    moe_inter_dim: int = 512
    n_layers: int = 12
    n_dense_layers: int = 3
    n_heads: int = 12

    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 4
    n_limited_groups: int = 2
    route_scale: float = 1.
    use_noaux_tc: bool = True
    bias_update_speed: float = 0.001
    use_seq_aux: bool = True
    seq_aux_alpha: float = 0.0001

    # mla
    q_lora_rank: int = 384  # 源码中若 q_lora_rank=0, 则不使用下投影，这里我们使用下投影，并略去若 q_lora_rank=0 的逻辑
    kv_lora_rank: int = 256  # 论文中 d_c = 4 * d_h
    qk_nope_head_dim: int = 64  # d_h
    qk_rope_head_dim: int = 32  # 论文中 d_h^R = d_h / 2
    v_head_dim: int = 64

    # RoPE
    rope_theta: float = 10000.0

    # MTP
    use_mtp: bool = True
    mtp_loss_lambda: float = 0.0001


# -------------------------------------------【RMSNorm】------------------------------------------- #
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    Args:
        dim (int): 嵌入维度
        eps (float): Epsilon 值用于确保数值稳定性, 默认为 1e-6
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim  # 在dim维度上进行归一化
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        RMSNorm 前向传播

        Args:
            x (torch.Tensor): 输入

        Returns:
            torch.Tensor: 归一化后的输出
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)  # 此函数需要 torch>=2.4.0


# ----------------------------------------------【RoPE】---------------------------------------------- #
def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    """
    计算频率矩阵, 并将其表示为复数的极坐标表示, 函数名中的 cis 指 cos(θ)+i·sin(θ), 表示一个复数位于单位圆上的位置

    Args:
        args (ModelArgs): 模型配置参数

    Returns:
        torch.Tensor: 预先计算的复数位置编码矩阵 (max_seq_len, qk_rope_head_dim//2)
    """
    dim = args.qk_rope_head_dim  # 解耦的 q_t^R 和 k_t^R 的每个头的维度，只有这两个部分需要应用 RoPE
    seqlen = args.max_seq_len  # 预计算长度为 max_seq_len 的位置编码矩阵
    base = args.rope_theta  # 旋转位置编码的基底【即 θ_d=b^(-2d/D) 中的 b】

    # 计算旋转矩阵中的不同维度的 θ_d 值, 即 θ_d=b^(-2d/D), [θ_0, θ_1, θ_2, ..., θ_{d-1}], 序列长度为 dim//2
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    t = torch.arange(seqlen)  # 生成位置索引 [0, 1, 2, ..., seqlen-1], 序列长度为 seqlen
    # torch.outer用于计算外积, 得到不同位置 m 和不同 θ_d 值的所有组合 m*θ_d
    # 得到的freqs矩阵形状为(seqlen, dim//2)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 生成一个模长为 1, 幅角为 freqs 的复数矩阵
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将 RoPE 应用于输入的张量

    Args:
        x (torch.Tensor): 需要应用 RoPE 的输入张量 (batch_size, seq_len, n_heads, qk_rope_head_dim)
        freqs_cis (torch.Tensor): 预先计算的复数位置编码矩阵 (seq_len, qk_rope_head_dim//2) 【seq_len 即就是 max_seq_len】

    Returns:
        torch.Tensor: 应用了 RoPE 的输出张量 (batch_size, seq_len, n_heads, qk_rope_head_dim)
    """
    dtype = x.dtype  # 记录 x 的数据类型
    # 将 head_dim 维度进行变换 (batch_size, seq_len, n_heads, qk_rope_head_dim) -> (batch_size, seq_len, n_heads, qk_rope_head_dim//2, 2)
    # 然后将两两维度变换为复数，形状进一步变为 (batch_size, seq_len, n_heads, qk_rope_head_dim//2)
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))  # x.float()会转换为 float32
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))  # (seq_len, qk_rope_head_dim//2) -> (1, seq_len, 1, qk_rope_head_dim//2)
    y = torch.view_as_real(x * freqs_cis).flatten(3)  # 逐元素相乘，freqs_cis 能够自动广播，相乘之后转换为实数形式，恢复成输入时的形状
    return y.to(dtype) # 恢复原输入的数据类型


# ----------------------------------------------【多头潜在注意力 MLA】---------------------------------------------- #
class MLA(nn.Module):
    """
    多头潜在注意力: Multi-Headed Latent Attention Layer (MLA)

    Attributes:
        dim (int): 嵌入维度【对应论文中的 d】
        n_heads (int): 注意力头数【对应论文中的 n_h】
        q_lora_rank (int): query 的下投影维度【对应论文中的 d_c'】
        kv_lora_rank (int): key/value 的下投影维度【对应论文中的 d_c】
        qk_nope_head_dim (int): 没有位置编码的 q_t^C 和 k_t^C 的每个头的维度【对应论文中的 d_h】
        qk_rope_head_dim (int): 解耦的带 RoPE 的 q_t^R 和 k_t^R 的每个头的维度【对应论文中的 d_h^R】
        qk_head_dim (int): 最终执行注意力计算的 query 和 key 的每个头的维度【即 d_h + d_h^R】
        v_head_dim (int): value 的每个头的维度, 可以与 qk_nope_head_dim 不同【但在论文中也同样设定为 d_h】
        softmax_scale (float): 注意力计算的缩放因子【即 1/sqrt(d_h + d_h^R)】
    """
    def __init__(self, args: DeepSeekV3ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads

        # 定义 query/key/value 维度
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # 低秩压缩 query
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)  # 下投影: d -> d_c'
        self.q_norm = RMSNorm(self.q_lora_rank)  # 下投影后对潜在向量进行一次 RMSNorm，原文中似乎没提到，但源码中有
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)  # 同时进行上投影 + 解耦多头 query 投影: d_c' -> (d_h + d_h^R) * n_h

        # key / value 的维度变换
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)  # 同时进行下投影 + 解耦共享 key 投影: d -> d_c + d_h^R
        self.kv_norm = RMSNorm(self.kv_lora_rank)  # 对潜在向量进行 RMSNorm
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)  # 同时进行 key 和 value 的上投影: d_c -> (d_h + d_h) * n_h

        # 输出的维度变换: d_h * n_h -> d
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        self.softmax_scale = self.qk_head_dim ** -0.5  # 注意力缩放因子
        # 论文中的低秩压缩的方式，缓存 c_t^KV 和 k_t^R
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        MLA 的前向传播

        Args:
            x (torch.Tensor): 输入 (batch_size, seq_len, dim)
            start_pos (int): 用于指定当前推理步骤的起始位置，即从序列的哪个位置开始计算
            freqs_cis (torch.Tensor): 预先计算的复数 RoPE 矩阵
            mask (Optional[torch.Tensor]): 掩码

        Returns:
            torch.Tensor: 输出 (batch_size, seq_len, dim)
        """

        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # -------------------------- query 部分 --------------------------
        # 同步计算 q_t^C 和 q_t^R
        q = self.wq_b(self.q_norm(self.wq_a(x)))  # (batch_size, seq_len, n_heads * qk_head_dim)
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)  # 划分多头: (batch_size, seq_len, n_heads, qk_head_dim)
        # 将 q 拆分成不带位置编码的 q_nope 部分和带位置编码的 q_pe 部分
        # q_nope: (batch_size, seq_len, n_heads, qk_nope_head_dim)
        # q_pe: (batch_size, seq_len, n_heads, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)  # 对解耦部分应用旋转位置编码

        # ----------------------- key / value 部分 -----------------------
        # 同步计算 k_t^C 和 k_t^R
        kv = self.wkv_a(x)  # 同时进行低秩压缩和解耦 key 的变换 (batch_size, seq_len, kv_lora_rank + qk_rope_head_dim)
        # 将上述结果拆分成潜在向量 kv 部分和带位置编码的 k_pe 部分
        # kv: (batch_size, seq_len, kv_lora_rank)
        # k_pe: (batch_size, seq_len, qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # 先增加 head 维度 (batch_size, seq_len, 1, qk_rope_head_dim)，而后对解耦部分应用旋转位置编码

        # --------------------------- 矩阵吸收 ---------------------------
        wkv_b = self.wkv_b.weight  # weight 的形状为(out_features, in_features), 即(n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
        wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)  # (n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
        # q_nope: (batch_size, seq_len, n_heads, qk_nope_head_dim)
        # wkv_b 截取的形状为: (n_heads, qk_nope_head_dim, kv_lora_rank), 即每个头的权重形状是(qk_nope_head_dim, kv_lora_rank)
        # 新的 q_nope 计算结果为: (batch_size, seq_len, n_heads, kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])  # 吸收 wkv_b TODO:能否直接初始化吸收好的权重矩阵

        # -------------------------- 注意力计算 --------------------------
        if self.training:  # 训练阶段不使用缓存
            kv = self.kv_norm(kv)
            k_pe = k_pe.squeeze(2)

            scores = (torch.einsum("bshc,btc->bsht", q_nope, kv) + torch.einsum("bshr,btr->bsht", q_pe, k_pe)) * self.softmax_scale

        else:  # 推理阶段使用缓存
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)  # 执行 RMSNorm 后，缓存 c_t^KV
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)  # 刚才为了应用 RoPE，为 k_pe 增加了 head 维度，现在取消 head 维度，然后缓存 k_t^R
            # 计算 nope 和 rope 部分的注意力
            # nope 部分: (batch_size, seq_len, n_heads, kv_lora_rank) (batch_size, cache_len, kv_lora_rank) -> (batch_size, seq_len, n_heads, cache_len)
            # rope 部分: (batch_size, seq_len, n_heads, qk_rope_head_dim) (batch_size, cache_len, qk_rope_head_dim) -> (batch_size, seq_len, n_heads, cache_len)
            # scores 的含义: (batch_size, seq_len, n_heads, cache_len) 每个位置、每个头对每个缓存历史位置的注意力得分
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) + 
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
            
        # 应用 mask
        if mask is not None:
            scores += mask.unsqueeze(1)  # mask 由 (seq_len, seq_len) 变为 (seq_len, 1, seq_len)，会自动在 batch 维度上广播
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)  # softmax 操作对数值稳定性要求较高，因此在 float32 下进行计算以避免溢出或下溢，然后再转换为 bf16
        
        # ----------------------- 计算输出+矩阵吸收 -----------------------
        # (batch_size, seq_len, n_heads, cache_len) (batch_size, cache_len, kv_lora_rank) -> (batch_size, seq_len, n_heads, kv_lora_rank)
        if self.training:
            x = torch.einsum("bsht,btc->bshc", scores, kv)
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        # (batch_size, seq_len, n_heads, kv_lora_rank) (n_heads, v_head_dim, kv_lora_rank) -> (batch_size, seq_len, n_heads, v_head_dim)
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        # (batch_size, seq_len, n_heads, v_head_dim) -> (batch_size, seq_len, n_heads * v_head_dim) -> (batch_size, seq_len, dim)
        x = self.wo(x.flatten(2))
        return x


# ----------------------------------------------【MoE部分】---------------------------------------------- #
class MLP(nn.Module):
    """
    前馈层, 前馈方法与Llama3相同, 用于创建共享专家

    Attributes:
        w1 (nn.Module): 实现 input-to-hidden 的转换
        w2 (nn.Module): 实现 hidden-to-output 的转换
        w3 (nn.Module): 实现 input-to-hidden 的转换
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        MLP 初始化

        Args:
            dim (int): 嵌入维度
            inter_dim (int): 隐藏层维度
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP 前向传播

        Args:
            x (torch.Tensor): 输入张量 (batch_size, seq_len, dim)

        Returns:
            torch.Tensor: 输出张量 (batch_size, seq_len, dim)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    即 Router, MoE 中的门控网口, 用于动态路由, 整体过程是: 
    1. 对专家进行分组, 共 n_groups 个组
    2. 每个组计算 2 个最大亲和度得分之和
    3. 根据上述结果选出 topk_groups 个组
    4. 从上述 topk_groups 个组的所有专家中, 选出 topk 个专家

    Attributes:
        dim (int): 嵌入维度
        topk (int): 每个输入激活的专家数量
        n_groups (int): 专家的分组数量
        topk_groups (int): 选中 topk 个分组
        route_scale (float): 路由权重的缩放因子
        n_routed_experts (int): 路由的专家数量
        weight (torch.nn.Parameter): 门控的可学习权重参数
        bias (Optional[torch.nn.Parameter]): 门控的偏置项
        original_scores (Optional[torch.Tensor]): 原始的亲和度得分, 形状为 (batch_size * seq_len, n_routed_experts)
    """
    def __init__(self, args: DeepSeekV3ModelArgs):
        """
        门控网络初始化

        Args:
            args (ModelArgs): 模型配置参数
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.route_scale = args.route_scale
        self.n_routed_experts = args.n_routed_experts
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))  # torch.empty 用于创建未初始化的张量，需要手动初始化
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts), requires_grad=False) if args.use_noaux_tc else None  # 用于无辅助损失负载均衡策略的 bias，不参与梯度计算，基于策略来更新bias，可以理解为通过策略干预而不是 loss 来进行更新的模型参数
        self.original_scores = None  # 用于存储原始的亲和度得分, 形状为 (batch_size * seq_len, n_routed_experts)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化参数
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)  # 这里将 bias 初始化为 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        门控网络的前向传播

        Args:
            x (torch.Tensor): 输入张量 (batch_size * seq_len, dim), 在输入前已经调整好形状

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 路由权重和选择的专家索引, 形状均为 (batch_size * seq_len, topk)
        """
        scores_logits = F.linear(x, self.weight, None)  # 计算所有 token 对专家的亲和度得分 (batch_size * seq_len, n_routed_experts)
        scores = scores_logits.sigmoid()  # 使用 sigmoid 来计算亲和度得分 (batch_size * seq_len, n_routed_experts)
        self.original_scores = scores  # 保留原始得分，用于后续根据原始得分抽取 topk 个专家 (batch_size * seq_len, n_routed_experts)
        scores_for_topk = scores # 创建一个新变量，避免原地操作
        
        if self.bias is not None:
            scores_for_topk = scores_for_topk + self.bias
        
        # 为专家分组
        if self.n_groups > 1:
            scores_view = scores_for_topk.view(x.size(0), self.n_groups, -1)  # (batch_size * seq_len, n_groups, n_routed_experts_per_group)
            if self.bias is None:
                group_scores = scores_view.amax(dim=-1)  # 取每个组的得分最大值，形状变为 (batch_size * seq_len, n_groups)
            else:
                group_scores = scores_view.topk(2, dim=-1)[0].sum(dim=-1)  # 取每个组最大两个得分的和，形状变为 (batch_size * seq_len, n_groups)
            indices_groups = group_scores.topk(self.topk_groups, dim=-1)[1]  # 获得组得分在前 topk_groups 的索引 (batch_size * seq_len, topk_groups)
            mask = torch.ones(x.size(0), self.n_groups, dtype=torch.bool, device=x.device)
            mask.scatter_(dim=1, index=indices_groups, value=False)  # 生成 mask 用于标记哪些组被选中，这里选中为 False (batch_size * seq_len, n_groups)
            scores_for_topk = scores_view.masked_fill(mask=mask.unsqueeze(-1), value=float("-inf")).flatten(1)  # 将未选中的，也就是为 True 组的所有专家的得分置为 -inf
        
        # 选出 topk 个专家，得到其索引和权重
        _, indices = torch.topk(scores_for_topk, self.topk, dim=-1)  # 在所选到的 topk_groups 个组的所有专家中，选出 topk 个专家的索引 (batch_size * seq_len, topk)
        weights = scores.gather(dim=1, index=indices)  # 从原始分数中按选出的索引抽取出亲和度得分，即权重 (batch_size * seq_len, topk)
        
        weights_sum = weights.sum(dim=-1, keepdim=True)
        weights = weights / (weights_sum + 1e-6)  # 将权重归一化【注意，这是在选出的 topk 中进行归一化】
        weights = weights * self.route_scale  # 应用缩放因子
            
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    MoE 中的路由专家, 它实际上和 MLP 结构相同

    Attributes:
        w1 (nn.Module): 实现 input-to-hidden 的转换
        w2 (nn.Module): 实现 hidden-to-output 的转换
        w3 (nn.Module): 实现 input-to-hidden 的转换
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化路由专家

        Args:
            dim (int): 嵌入维度
            inter_dim (int): 隐藏层维度
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        路由专家前向传播

        Args:
            x (torch.Tensor): 输入张量 (batch_size, seq_len, dim)

        Returns:
            torch.Tensor: 输出张量 (batch_size, seq_len, dim)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) 混合专家模块

    Attributes:
        dim (int): 嵌入维度
        n_routed_experts (int): 路由专家数量
        n_activated_experts (int): 每个输入激活的专家数
        gate (nn.Module): 门控机制
        experts (nn.ModuleList): 专家列表
        shared_experts (nn.Module): 共享专家
        use_seq_aux (bool): 是否使用序列级别的辅助损失
        seq_aux_alpha (float): 序列级别的辅助损失的权重
        bias_update_speed (float): 偏置更新速度
    """
    def __init__(self, args: DeepSeekV3ModelArgs):
        """
        MoE 初始化

        Args:
            args (ModelArgs): 模型配置参数
        """
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for _ in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
        self.use_seq_aux = args.use_seq_aux
        self.seq_aux_alpha = args.seq_aux_alpha
        self.bias_update_speed = args.bias_update_speed  # 用于无辅助损失负载均衡策略的 bias 的更新速度

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        MoE 前向传播

        Args:
            x (torch.Tensor): 输入张量 (batch_size, seq_len, dim)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: 输出张量 (batch_size, seq_len, dim), 序列级辅助损失, 全局负载情况
        """
        shape = x.size()
        bsz, seqlen = shape[:2]
        x = x.view(-1, self.dim)  # 重新划分形状为 (batch_size * seq_len, dim)
        weights, indices = self.gate(x)  # 计算得到 x 的路由权重和选择的专家索引，形状均为 (batch_size * seq_len, topk)
        y = torch.zeros_like(x)  # (batch_size * seq_len, dim)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)  # bincount 用于计算非负整数张量中每个值的出现次数，即此列表保存了一个batch里每个专家对应的激活次数
        global_counts = None  # 初始化全局 counts

        # -------------------- 无辅助损失负载均衡策略 --------------------
        # 这里我们计算出了一个 batch 中所有专家的激活情况，故顺便在此应用无辅助损失的负载均衡策略来更新 gate 中的 bias
        # 每一个 MoE 层更新自己的 Gate 的 bias，下一个 batch 的数据将使用更新的 bias，训练的最后一组数据更新完 bias 后，将作为模型参数保存下来
        if self.gate.bias is not None and self.training:
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
            scores_for_seq_aux = self.gate.original_scores.view(bsz, seqlen, -1)  # 此即原始的 s_{i,t} (batch_size, seq_len, n_routed_experts)
            scores_for_seq_aux = scores_for_seq_aux / scores_for_seq_aux.sum(dim=-1, keepdim=True)  # 沿着 n_routed_experts 的方向归一化，形成 s_{i,t}'
            P_i = scores_for_seq_aux.mean(dim=1)  # 沿着 token 的方向求平均 (batch_size, n_routed_experts)

            # 计算 f_i，含义为第 i 个专家在每个 token 上的平均激活次数
            # indices 计算了一个 batch 中每个 token 激活了哪些专家，现在要计算每个序列中，每个专家被哪些 token 激活
            # 可以使用 one-hot 编码来快速计算每个专家被多少个 token 激活
            f_i = F.one_hot(indices.view(bsz, -1), num_classes=self.n_routed_experts)  # (batch_size, seq_len * topk, n_routed_experts)
            f_i = f_i.sum(dim=1)  # 沿 seq_len * topk 维度相加后，求出每个专家被多少个 token 激活 (bsz, n_routed_experts)
            f_i = (f_i * self.n_routed_experts) / (self.n_activated_experts * seqlen)  # 计算每个专家的平均激活次数并乘以系数 (batch_size, n_routed_experts)

            seq_aux_loss = (f_i * P_i).sum() * self.seq_aux_alpha  # 计算序列级别的辅助损失
        else:
            seq_aux_loss = None  # 不使用序列级别的辅助损失时，将损失设为 None

        # 为每个 token 计算路由专家的输出和
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)  # 找到激活了第 i 个专家的 token，idx 代表行索引(即第几个 token)，top 代表列索引(即该 token 的 top 几选择)，idx 和 top 的类型为 torch.Tensor
            # 假设 n_matches 是匹配当前专家的 token 数量，那么 expert(x[idx]) 的形状是 (n_matches, dim)，weights[idx, top] 的形状是 (n_matches,)
            # None 用于增加一个维度，使形状变为 (n_matches, 1)
            y[idx] += expert(x[idx]) * weights[idx, top, None]  # 如果 idx 是标量(如 idx=5 )，结果形状是 (dim,)；如果 idx 是单元素张量(如 idx=tensor([5]) )，结果形状是 (1, dim)
        
        # 计算共享专家的输出
        z = self.shared_experts(x)  # (batch_size * seq_len, dim)
 
        return (y + z).view(shape), seq_aux_loss, global_counts


# ----------------------------------------------【Transformer Block部分】---------------------------------------------- #
class Block(nn.Module):
    """
    Transformer Block, 包括 Attention 和 Feed-Forward 部分

    Attributes:
        attn (nn.Module): 注意力层 (MLA)
        ffn (nn.Module): 前馈网络 (MoE)
        attn_norm (nn.Module): 注意力层的 Layer Normalization
        ffn_norm (nn.Module): 前馈网络的 Layer normalization
    """
    def __init__(self, layer_id: int, args: DeepSeekV3ModelArgs):
        """
        初始化 Transformer Block

        Args:
            layer_id (int): Transformer 的层索引
            args (ModelArgs): 模型配置参数
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args) # 前几层是常规的 MLP 层
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Transformer Block 的前向传播

        Args:
            x (torch.Tensor): 输入 (batch_size, seq_len, dim)
            start_pos (int): 用于指定当前推理步骤的起始位置，即从序列的哪个位置开始计算
            freqs_cis (torch.Tensor): 预先计算的复数 RoPE 矩阵
            mask (Optional[torch.Tensor]): 掩码

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 输出 (batch_size, seq_len, dim)
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)  # 残差连接
        if isinstance(self.ffn, MoE):
            h, seq_aux_loss, global_counts = self.ffn(self.ffn_norm(x))  # 如果是 MoE 层，返回输出和 seq_aux_loss
            return x + h, seq_aux_loss, global_counts
        x = x + self.ffn(self.ffn_norm(x))  # 如果是非 MoE 层，seq_aux_loss 为 None
        return x, None, None


# ----------------------------------------------【MTP部分】---------------------------------------------- #
class MTP(nn.Module):
    """
    多token预测(Multi-Token Prediction, MTP)

    Attributes:
        args (ModelArgs): 模型配置参数
        embed (nn.Module): 嵌入层
        head (nn.Module): 输出投影
        h_norm (nn.Module): 对上一个 MTP 模块或主模型输出的 hidden state 应用的 Layer Normalization
        x_norm (nn.Module): 对本 MTP 模块输入应用的 Layer Normalization
        output_norm (nn.Module): 对本 MTP 模块输出应用的 Layer Normalization
        linear_proj (nn.Module): 线性投影层，用于将 MTP 模块的两个输入拼接后进行投影
        transformer_block (nn.Module): Transformer Block
    """
    def __init__(self, args: DeepSeekV3ModelArgs, embed: nn.Module, head: nn.Module):
        """
        初始化 MTP

        Args:
            args (ModelArgs): 模型配置参数
            embed (nn.Module): 嵌入层, 来自于 Transformer 共享
            head (nn.Module): 输出投影, 来自于 Transformer 共享
        """
        super().__init__()
        self.embed = embed
        self.head = head
        self.h_norm = RMSNorm(args.dim)
        self.x_norm = RMSNorm(args.dim)
        self.output_norm = RMSNorm(args.dim)  # 原文中的图中在每个 MTP 的 Transformer Block 后没有进行 norm，但主模型有，这里也加上
        self.linear_proj = nn.Linear(2 * args.dim, args.dim, bias=False)
        self.transformer_block = Block(0, args)  # 原文中未说明 MTP 使用的是否是 MoE 架构，这里使用普通 MLP 作为前馈
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MTP 的前向传播

        Args:
            x (torch.Tensor): 输入 token_ids (batch_size, mtp_seq_len)
            h (torch.Tensor): 上一个 MTP 模块或主模型的输出 (batch_size, mtp_seq_len, dim)
            start_pos (int): 用于指定当前推理步骤的起始位置，即从序列的哪个位置开始计算
            freqs_cis (torch.Tensor): 预先计算的复数 RoPE 矩阵

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输出 Logits (batch_size, vocab_size) 和给下一个 MTP 使用的 h (batch_size, mtp_seq_len, dim)
        """
        seqlen = x.size(1)
        mask = None

        x = self.embed(x)  # (batch_size, mtp_seq_len, dim)
        x = self.x_norm(x)
        h = self.h_norm(h)
        x = self.linear_proj(torch.cat([x, h], dim=-1))  # 拼接后进行投影 (batch_size, mtp_seq_len, dim)

        if seqlen > 1:  # seqlen 为 1 通常是使用 KV Cache 推理时，若大于 1 ，则需要使用因果 mask
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device).triu_(1)  # 下三角(含对角线)设置为 0
        x, _, _ = self.transformer_block(x, start_pos, freqs_cis, mask)  # (batch_size, mtp_seq_len, dim)

        h = x  # 输出 h 作为下一个 MTP 模块的输入
        x = self.output_norm(x)  # (batch_size, mtp_seq_len, dim)
        logits = self.head(x)  # (batch_size, mtp_seq_len, vocab_size)

        return logits, h


# ----------------------------------------------【整体模型部分】---------------------------------------------- #
class DeepSeekV3Model(BaseModel):
    """
    Transformer 整体模型

    Attributes:
        max_seq_len (int): 最大序列长度
        embed (nn.Module): 嵌入层
        layers (torch.nn.ModuleList): Transformer Blocks 列表
        norm (nn.Module): 在最后一个 Transformer Block 后应用 Layer Normalization
        head (nn.Module): 输出投影
        freqs_cis (torch.Tensor): 预先计算的复数 RoPE 矩阵
        use_mtp (bool): 是否使用 MTP
        mtp_loss_lambda (float): MTP 损失的权重
        MTP (torch.nn.ModuleList): MTP 模块
    """
    model_name = "mini_deepseekv3"  # 模型名称

    def __init__(self, args: DeepSeekV3ModelArgs):
        """
        初始化 Transformer

        Args:
            args (ModelArgs): 模型配置参数
        """
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.use_mtp = args.use_mtp
        self.mtp_loss_lambda = args.mtp_loss_lambda
        self.mtp =MTP(args, self.embed, self.head)
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids (torch.Tensor): 输入张量, 内容为 token_ids (batch_size, seq_len)
            targets (torch.Tensor): 目标张量, 内容为 token_ids (batch_size, seq_len)
            start_pos (int): 起始位置, 默认为 0

        Returns:
            torch.Tensor: 输出 Logits (batch_size, vocab_size), Loss
        """
        # 变量初始化
        seqlen = input_ids.size(1)
        h = self.embed(input_ids)  # (batch_size, seq_len, dim)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        main_loss = 0.0
        total_seq_aux_loss = 0.0  # 用于存储所有 MoE 层的 seq_aux_loss 之和
        mtp_loss = 0.0  # 用于存储 MTP 的损失
        all_global_counts = []

        # ----------------------- 主模型部分 -----------------------
        # 对于一个原始样本数据 data，训练的输入 tokens 为 data[:-1]，targets 为 data[1:]
        # 主模型部分使用全部的 data[:-1] 做下一个 token 的预测，预测目标为 data[1:]
        if seqlen > 1:  # seqlen 为 1 通常是使用 KV Cache 推理时，若大于 1，则需要使用因果 mask
            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device).triu_(1)  # 下三角(含对角线)设置为 0
        
        for layer in self.layers:
            h, seq_aux_loss, global_counts = layer(h, start_pos, freqs_cis, mask)  # (batch_size, seq_len, dim)
            if seq_aux_loss is not None:  # 如果 seq_aux_loss 不为 None，说明是 MoE 层，需要累加
                total_seq_aux_loss += seq_aux_loss
            if global_counts is not None:  # 如果 global_counts 不为 None，说明是 MoE 层，需要累加
                all_global_counts.append(global_counts)
        
        h_for_mtp = h  # 分离主模型的输出，用于 MTP 输入 (batch_size, seq_len, dim)
        h = self.norm(h)  # (batch_size, seq_len, dim)
        logits = self.head(h)  # (batch_size, seq_len, vocab_size)
        if self.training:
            main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")

        # ----------------------- MTP 部分 -----------------------
        # 此处仅使用固定预测深度为 1 的逻辑，即只使用 1 个 MTP 模块
        # MTP 使用 data[:-2] 做下两个 token 的预测，预测目标为 data[2:]
        if self.use_mtp and self.training:
            mtp_logits, _ = self.mtp(input_ids[:, 1:], h_for_mtp[:, :-1], 0, self.freqs_cis[0:seqlen-1])  # (batch_size, seq_len - 1, vocab_size)
            targets_for_mtp = targets[:, 1:]  # (batch_size, seq_len - 1)
            mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, mtp_logits.size(-1)), targets_for_mtp.reshape(-1), reduction="mean")  # 计算 MTP 的损失
        
        # 计算总损失
        loss = main_loss + total_seq_aux_loss + self.mtp_loss_lambda * mtp_loss

        return logits, loss, (main_loss, total_seq_aux_loss, self.mtp_loss_lambda * mtp_loss, all_global_counts) # 第三部分用于给 tensorboard 提供需要记录的模型内部变量


if __name__ == "__main__":
    args = DeepSeekV3ModelArgs()

    # MLA模块测试
    print(f"{'-'*10} test MLA {'-'*10}")
    x = torch.randn(1, args.max_seq_len, args.dim)
    print(f"attention input size: {x.size()}")
    attention = MLA(args)
    freq_cis = precompute_freqs_cis(args)  # 预计算旋转矩阵
    attention_output = attention(x, start_pos=0, freqs_cis=freq_cis, mask=None)
    print(f"attention output size: {attention_output.size()}")

    # Gate模块梯度测试
    print(f"{'-'*10} test Gate {'-'*10}")
    x = torch.randn(args.max_batch_size*args.max_seq_len, args.dim)  # (batch_size * seq_len, dim)
    gate = Gate(args)
    # 前向传播
    weights, indices = gate(x)
    loss = weights.sum()  # 假设损失函数是 gate 的输出求和
    # 反向传播
    loss.backward()
    # 检查 bias 的梯度
    if gate.bias is not None:
        print('grad of bias: \n', gate.bias.grad)
    print('bias: \n', gate.bias)
    print('grad of weight: \n', gate.weight.grad)

    # MoE模块测试
    print(f"{'-'*10} test MoE {'-'*10}")
    x = torch.randn(1, args.max_seq_len, args.dim)  # (batch_size, seq_len, dim)
    print(f"moe input size: {x.size()}")
    moe = MoE(args)
    moe_output, seq_aux_loss = moe(x)  # (batch_size, seq_len, dim)
    print(f"moe output size: {moe_output.size()}")
    print(f"seq_aux_loss: {seq_aux_loss}")

    # 全模型测试
    print(f"{'-'*10} test Transformer {'-'*10}")
    x = torch.randint(0, args.vocab_size, (1, args.max_seq_len))  # (batch_size, seq_len)
    t = torch.randint(0, args.vocab_size, (1, args.max_seq_len))  # (batch_size, seq_len)
    print(f"transformer input size: {x.size()}")
    transformer = DeepSeekV3Model(args)
    transformer_output = transformer(x, t)  # (batch_size, vocab_size)
    print(f"transformer output size: {transformer_output[0].size()}")

    print(f'model _name: {transformer.model_name}')

