from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..cache import MiniQwen3NextDynamicCache
from ..base_module import RMSNormGated


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    为 hidden_states 应用 attention mask, 将填充位置的 hidden_states 设置为 0
    
    Args:
        hidden_states (torch.Tensor): 输入张量 (batch_size, seq_len, hidden_size)
        attention_mask (torch.Tensor | None): 掩码张量 (batch_size, seq_len)
    """
    # 仅在 attention_mask 存在且 batch_size 和 seq_len 均大于 1 时才计算
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    使用 conv1d 实现因果卷积的状态更新
    
    Args:
        hidden_states (torch.Tensor): 输入张量 (batch_size, key_dim + key_dim + value_dim, seq_len)
        conv_state (torch.Tensor): 卷积状态张量 (batch_size, key_dim + key_dim + value_dim, conv_kernel_size)
        weight (torch.Tensor): 卷积权重张量 (key_dim + key_dim + value_dim, conv_kernel_size)
        bias (torch.Tensor | None): 卷积偏置张量 (key_dim + key_dim + value_dim,)
    
    Returns:
        torch.Tensor: 卷积输出张量 (batch_size, key_dim + key_dim + value_dim, seq_len)
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)  # (batch_size, key_dim + key_dim + value_dim, seq_len + conv_kernel_size)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])  # 更新 conv_state
    
    # 计算卷积输出
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    
    return out


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6) -> torch.FloatTensor:
    """L2 Norm"""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    分块 Gated Delta Rule, 用于训练和 prefill 阶段的并行计算, 此时的 num_k_heads 已经经过 repeat 与 num_v_heads 相同, 本函数内均使用 num_heads 表示

    Args:
        query (torch.Tensor): 查询张量 (batch_size, seq_len, num_heads, head_k_dim)
        key (torch.Tensor): 键张量 (batch_size, seq_len, num_heads, head_k_dim)
        value (torch.Tensor): 值张量 (batch_size, seq_len, num_heads, head_v_dim)
        g (torch.Tensor): 遗忘门 log 值 (batch_size, seq_len, num_heads), exp(g) 为实际衰减系数
        beta (torch.Tensor): 写入门 (batch_size, seq_len, num_heads), 控制写入强度
        chunk_size (int): 分块大小, 默认 64
        initial_state (Optional[torch.Tensor]): 初始循环状态 (batch_size, num_heads, head_k_dim, head_v_dim)
        output_final_state (bool): 是否输出最终状态, 用于缓存
        use_qk_l2norm_in_kernel (bool): 是否对 query/key 做 L2 归一化

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: 
            - 注意力输出 (batch_size, seq_len, num_heads, head_v_dim)
            - 最终循环状态 (batch_size, num_heads, head_k_dim, head_v_dim), 若 output_final_state=False 则为 None
    """
    initial_dtype = query.dtype

    # 使用 L2 Norm
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # 将 seq_len 维度与 num_heads 维度互换，并转换为 fp32
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    
    # 变量准备，此时的 num_k_heads 已经经过 repeat 与 num_v_heads 相同，均使用 num_heads 表示
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size  # 计算最后一个 chunk 需要 pad 的数量
    
    # F.pad 的 padding 参数是从最后一维开始，成对指定的
    # (0, 0, 0, pad_size) 是指：最后一维 左边 pad 0，最后一维 右边 pad 0，倒数第2维 左边 pad 0，倒数第2维 右边 pad pad_size
    # 最终会在 seq_len 维度上 pad 0 到 chunk_size 的整数倍
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size  # pad 后的总长度
    scale = 1 / (query.shape[-1] ** 0.5)  # 缩放因子
    query = query * scale

    # 计算 beta 和 key/value 的乘积，用于后续进一步与下三角模块计算得到 W 和 U 矩阵
    v_beta = value * beta.unsqueeze(-1)  # (batch_size, num_heads, seq_len, head_v_dim)
    k_beta = key * beta.unsqueeze(-1)  # (batch_size, num_heads, seq_len, head_k_dim)
    
    # reshape 成 chunks
    # (batch_size, num_heads, num_chunks, chunk_size, head_k/v_dim)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)  # (batch_size, num_heads, num_chunks, chunk_size)
    # 创建包含主对角线的上三角矩阵，上三角及对角线为 True，下三角为 False
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # 计算 chunk 衰减系数
    # cumsum 用于计算累计和，例如 g = [g0, g1, g2, ...]，则 cumsum(g) = [g0, g0+g1, g0+g1+g2, ...]
    # α = exp(g)，因此首先通过 cumsum 计算得到 g 的累计和，后续可以快速通过 exp 计算得到 α 的累乘
    # g 在此时已经分成了 chunk，因此这里的累计和是从 chunk 的起点开始的
    g = g.cumsum(dim=-1)  # (batch_size, num_heads, num_chunks, chunk_size)
    # 这里的 decay_mask 就是论文 Gated DeltaNet 中的 Gamma (Γ)，即下三角衰减比值矩阵
    # 只看最后两维度，(g.unsqueeze(-1) - g.unsqueeze(-2)) 即 (..., chunk_size, 1) - (..., 1, chunk_size) = (..., chunk_size, chunk_size)，得到每个位置对其他各位置的相对累计和
    # 对该相对累计和取下三角，然后取 exp，就得到每个位置对先前位置的衰减比值，保证了因果
    # 注意在 exp 后还需要再进行一次 tril，因为 exp 会把第一次 tril 的上三角 0 变为 exp(0) = 1，因此需要再进行一次 tril 保证上三角为 0
    # 因此 decay_mask[i,j] 表示从位置 j 的写入会衰减到位置 i，且 i > j
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    # 我们在公式中下三角部分的推导结果为：strictLower(diag(β)·(Γ⊙(K·K^T)))
    # 其中 Γ 是给 K·K^T 的每个元素乘上对应的衰减比值，diag(β) 是给 K·K^T 的每行乘上对应的 β，他们都是以相乘的形式作用在 K·K^T 上的
    # 因此它实际上等价于 strictLower(Γ⊙(diag(β)·(K·K^T))) = strictLower(Γ⊙((diag(β)·K)·K^T))
    # 这里 k_beta 相当于提前吸收了 diag(β) 部分，因此直接计算 k_beta·K^T，再与 decay_mask 相乘即可
    # 最后，应用上三角的 mask，确保最终的 attn 矩阵为严格的下三角矩阵，对角线也为 0
    # 这里给 attn 前取负号，是为了后续的迭代求逆
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)  # (batch_size, num_heads, num_chunks, chunk_size, chunk_size)
    
    # 此循环是用于迭代求解 [I + strictLower(diag(β)·(Γ⊙(K·K^T)))] 的逆，具体原理请参考本人手撕 Qwen3-Next 的博客
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()  # (batch_size, num_heads, num_chunks, i)
        sub = attn[..., :i, :i].clone()  # (batch_size, num_heads, num_chunks, i, i)，i=1时，attn 的左上角子块就是 0，它就相当于已知的 $\widetilde{\mathbf{T}}$ 的左上角子块作为迭代初值
        # (row.unsqueeze(-1) * sub).sum(-2) 是 (..., i, 1) * (..., i, i) = (..., i, i) 然后 sum(-2) 得到 (..., i)，相当于对 sub 的行按照 row 进行加权求和
        # 它也相当于行向量 row.unsqueeze(-2) @ sub，即 (..., 1, i) @ (..., i, i) = (..., 1, i) 然后 squeeze(-2) 恢复形状得到 (..., i)
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    # (I + strictLower(diag(β)·(Γ⊙(K·K^T))))^(-1) @ (diag(β)·V) 由此得到 U 矩阵
    value = attn @ v_beta  # (batch_size, num_heads, num_chunks, chunk_size, head_v_dim)
    # 这里的 k_cumdecay 就是 W 矩阵，具体原理请参考本人手撕 Qwen3-Next 的博客
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))  # (batch_size, num_heads, num_chunks, chunk_size, head_k_dim)
    
    # 设置初始化状态 (batch_size, num_heads, head_k_dim, head_v_dim)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    # 初始化输出张量
    core_attn_out = torch.zeros_like(value)  # (batch_size, num_heads, num_chunks, chunk_size, head_v_dim)
    # 创建一个不包含主对角线的上三角矩阵（严格上三角），上三角为 True，下三角及对角线为 False
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # 逐 chunk 计算状态和输出
    for i in range(0, total_sequence_length // chunk_size):
        # 获取 chunk 内 QKV
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]  # (batch_size, num_heads, chunk_size, head_k/v_dim)
        # NOTE: QK 需要的是 decay mask，Gated DeltaNet 原文中写的 M 是普通因果 mask，似乎是笔误，但代码这里没问题
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)  # (batch_size, num_heads, chunk_size, chunk_size)
        
        # 计算等效 value
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state  # 计算 WS^T，相当于 v_old (batch_size, num_heads, chunk_size, head_v_dim)
        v_new = v_i - v_prime  # U - WS^T，相当于需要写入的 v_new，即等效 value (batch_size, num_heads, chunk_size, head_v_dim)
        
        # 计算 chunk 的输出 O
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state  # 计算 QS^T，其中 Q 是衰减到 chunk 起点的 (batch_size, num_heads, chunk_size, head_v_dim)
        core_attn_out[:, :, i] = attn_inter + attn @ v_new  # chunk 输出 (batch_size, num_heads, chunk_size, head_v_dim)
        
        # 计算 chunk 的状态 S (batch_size, num_heads, head_k_dim, head_v_dim)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()  # 取 g 在 chunk_size 上的最后一个值，并取 exp，即 \gamma^C，使得 S 衰减到 chunk 终点
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new  # 这里对 K 进行 \gamma^i 到 \gamma^C 的衰减
        )

    if not output_final_state:
        last_recurrent_state = None
    # 恢复形状为 (batch_size, num_heads, total_seq_len, head_v_dim)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]  # 裁剪有效输出，去掉 pad 部分
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)  # (batch_size, seq_len, num_heads, head_v_dim)
    
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    递归 Gated Delta Rule, 用于 decode 阶段的逐步计算
    
    Args:
        query (torch.Tensor): 查询张量 (batch_size, seq_len, num_heads, head_k_dim)
        key (torch.Tensor): 键张量 (batch_size, seq_len, num_heads, head_k_dim)
        value (torch.Tensor): 值张量 (batch_size, seq_len, num_heads, head_v_dim)
        g (torch.Tensor): 遗忘门 log 值 (batch_size, seq_len, num_heads), exp(g) 为实际衰减系数
        beta (torch.Tensor): 写入门 (batch_size, seq_len, num_heads), 控制写入强度
        initial_state (Optional[torch.Tensor]): 初始循环状态 (batch_size, num_heads, head_k_dim, head_v_dim)
        output_final_state (bool): 是否输出最终状态, 用于缓存
        use_qk_l2norm_in_kernel (bool): 是否对 query/key 做 L2 归一化

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: 
            - 注意力输出 (batch_size, seq_len, num_heads, head_v_dim)
            - 最终循环状态 (batch_size, num_heads, head_k_dim, head_v_dim), 若 output_final_state=False 则为 None

    """
    initial_dtype = query.dtype
    
    # 使用 L2 Norm
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # 将 seq_len 维度与 num_heads 维度互换，并转换为 fp32
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    # 变量准备，此时的 num_k_heads 已经经过 repeat 与 num_v_heads 相同，均使用 num_heads 表示
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    # 初始化输出张量和循环状态
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)  # (batch_size, num_heads, seq_len, head_v_dim)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)  # (batch_size, num_heads, head_k_dim, head_v_dim)
        if initial_state is None
        else initial_state.to(value)
    )

    # 逐时间步计算状态和输出
    for i in range(sequence_length):
        q_t = query[:, :, i]  # (batch_size, num_heads, head_k_dim)
        k_t = key[:, :, i]  # (batch_size, num_heads, head_k_dim)
        v_t = value[:, :, i]  # (batch_size, num_heads, head_v_dim)
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # (batch_size, num_heads, 1, 1)
        beta_t = beta[:, :, i].unsqueeze(-1)  # (batch_size, num_heads, 1)

        # 这里的计算过程与经典公式相比，稍微进行了一点变换，具体见本人手撕 Qwen3-Next 的博客
        last_recurrent_state = last_recurrent_state * g_t  # 将 α 吸收进状态中，后续过程可以看作是 DeltaNet (batch_size, num_heads, head_k_dim, head_v_dim)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)  # 相当于S_{t-1}·k，得到 v_old (batch_size, num_heads, head_v_dim)
        delta = (v_t - kv_mem) * beta_t  # 得到 β(v - v_old) (batch_size, num_heads, head_v_dim)
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # S_t = S_{t-1} + β(v - v_old)·k^T (batch_size, num_heads, head_k_dim, head_v_dim)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)  # S_t·q，得到最终输出 (batch_size, num_heads, head_v_dim)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)  # (batch_size, seq_len, num_heads, head_v_dim)
    
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    """
    Gated Delta Net (GDN)

    Args:
        layer_idx (int): 层索引
        hidden_size (int): 隐状态维度
        num_k_heads (int): key 头数
        num_v_heads (int): value 头数
        head_k_dim (int): key 每个头的维度
        head_v_dim (int): value 每个头的维度
        conv_kernel_size (int): 卷积核大小
        layer_norm_epsilon (float): 层归一化 epsilon
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        layer_norm_epsilon: float,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads  # q,k 头数
        self.num_v_heads = num_v_heads  # v,z 头数
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim  # 要求 num_v_heads % num_k_heads == 0
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.layer_norm_epsilon = layer_norm_epsilon

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 定义卷积通道数，query/key/value 均需要做 short conv
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,  # 输入通道数
            out_channels=self.conv_dim,  # 输出通道数
            bias=False,
            kernel_size=self.conv_kernel_size,  # 卷积核大小
            groups=self.conv_dim,  # 每个通道进行独立的卷积操作，因此不存在跨通道信息交换
            padding=self.conv_kernel_size - 1,  # 在序列两侧各 padding k-1 个 0，输入长度变为 L + 2*(k-1)，卷积后输出长度变为 L + k - 1，然后会截断为 L
        )
        
        # 输入投影
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2  # q, k, v, z 的总投影维度，z 是门控中的线性层
        projection_size_ba = self.num_v_heads * 2  # Gated Delta Rule 的 α β 的总投影维度
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)
        
        # 时间偏置：(num_v_heads,) 每个 head 一个标量，初始化为 1
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        # 衰减基数：首先创建一个 (num_v_heads,) 的空张量，然后从 [0,16) 中均匀采样原地写入
        # 它让不同 head 有不同的初始衰减速率，[0,16) 是经验值
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        # A 必须保持为正数，如果直接把 A 设置为可学习参数，梯度更新可能把它推成负数
        # 因此这里参数化时使用 log(A)，forward 时再通过 exp 恢复 A
        self.A_log = nn.Parameter(torch.log(A))

        # 输出投影前的 RMSNorm + Gate，图中是 Zero-Centered RMSNorm，但源码中是常规的 RMSNorm
        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_update = torch_causal_conv1d_update
        self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        从 mixed_qkvz 和 mixed_ba 中派生出 query、key、value、z、b、a 张量
        
        Args:
            mixed_qkvz (torch.Tensor): 投影后的 qkvz 张量 (batch_size, seq_len, key_dim*2 + value_dim*2)
            mixed_ba (torch.Tensor): 投影后的 ba 张量 (batch_size, seq_len, num_v_heads*2)
        """
        # NOTE：
        # 本函数首先将 mixed_qkvz 和 mixed_ba 按照 num_k_heads view 成 4D，再在每段内 split
        # 并没有直接把 3D 的最后一维按 [key_dim, key_dim, value_dim, value_dim] 或 [num_v_heads, num_v_heads] 切分，然后再 reshape
        # 
        # 个人猜测：
        # Qwen3-Next 在他们源码中，是按照逐 k head 排布的方式训练得到的权重
        # 在实现 transformers 版本时，为了兼容模型权重，于是添加此函数用于按训练实现约定的布局去解码
        # 如果我们是重新开始训练模型，这里直接拆分维度，再各自 reshape 成想要的形状即可，但为了尽可能与 transformers 版本保持一致，这里不做修改

        # 新的 qkvz 形状: (batch_size, seq_len, num_k_heads, 2*head_k_dim + 2*num_v_heads/num_k_heads*head_v_dim)
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        # 新的 ba 形状: (batch_size, seq_len, num_k_heads, 2*num_v_heads/num_k_heads)
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        
        # 拆分 qkvz 和 ba
        # q/k 形状为 (batch_size, seq_len, num_k_heads, head_k_dim)
        # v/z 形状为 (batch_size, seq_len, num_k_heads, num_v_heads/num_k_heads*head_v_dim)
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        # b/a 形状为 (batch_size, seq_len, num_k_heads, num_v_heads/num_k_heads)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)  # (batch_size, seq_len, num_v_heads, head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)  # (batch_size, seq_len, num_v_heads, head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)  # (batch_size, seq_len, num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)  # (batch_size, seq_len, num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[MiniQwen3NextDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        # 用于判断当前是 prefill 阶段还是 decode 阶段
        use_precomputed_states = (
            cache_params is not None  # 缓存参数不为空
            and cache_params.has_previous_state  # 存在之前的缓存状态
            and seq_len == 1  # 输入序列长度为 1
            and cache_position is not None  # 缓存位置不为空
        )

        # 获取卷积状态和循环状态
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]  # (batch_size, key_dim + key_dim + value_dim, conv_kernel_size)
            recurrent_state = cache_params.recurrent_states[self.layer_idx]  # (batch_size, key_dim + key_dim + value_dim, conv_kernel_size)

        # ------------------------- 1. 输入投影 -------------------------
        # 最终得到的形状为：
        # query: (batch_size, seq_len, key_dim)
        # key: (batch_size, seq_len, key_dim)
        # value: (batch_size, seq_len, value_dim)
        # z: (batch_size, seq_len, num_v_heads, head_v_dim)
        # b: (batch_size, seq_len, num_v_heads)
        # a: (batch_size, seq_len, num_v_heads)
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)  # (batch_size, seq_len, key_dim*2 + value_dim*2)
        projected_states_ba = self.in_proj_ba(hidden_states)  # (batch_size, seq_len, num_v_heads*2)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))  # (batch_size, seq_len, key_dim/value_dim)

        mixed_qkv = torch.cat((query, key, value), dim=-1)  # (batch_size, seq_len, key_dim + key_dim + value_dim)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (batch_size, key_dim + key_dim + value_dim, seq_len)

        # ------------------------- 2. qkv 卷积 -------------------------
        if use_precomputed_states:
            # decode 阶段
            # 利用之前的 conv_state 计算新卷积输出，并更新 conv_state
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,  # (batch_size, key_dim + key_dim + value_dim, seq_len)
                conv_state,  # (batch_size, key_dim + key_dim + value_dim, conv_kernel_size)
                # conv1d 的权重形状为 (out_channels, in_chennels / groups, kernel_size)，由于这里是通道分离卷积，因此 in_channels / groups = 1
                self.conv1d.weight.squeeze(1),  # (key_dim + key_dim + value_dim, conv_kernel_size)
                self.conv1d.bias,  # (key_dim + key_dim + value_dim,)
            )
        else:
            # prefill 阶段
            if cache_params is not None:
                # 初始化卷积缓存状态，在最后一个维度，也就是序列长度维度的左侧，填充 conv_kernel_size - seq_len 个 0
                # 当 seq_len  >= conv_kernel_size 时，conv_kernel_size - seq_len 为负数或 0，此时会截断张量，只保留最后 conv_kernel_size 个时间步
                # 当 seq_len < conv_kernel_size 时，左侧填充 0，使总长度达到 conv_kernel_size
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))  # (batch_size, key_dim + key_dim + value_dim, conv_kernel_size)
                cache_params.conv_states[self.layer_idx] = conv_state  # 注意这里 conv_state 保存的是卷积前的状态
            # 每个通道进行独立的卷积，因此 qkv 可以拼接后一起执行，卷积输出后，将多余的 k-1 部分截断
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        # ------------------------- 3. 形状调整 -------------------------
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)  # (batch_size, seq_len, num_k_heads, head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)  # (batch_size, seq_len, num_k_heads, head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)  # (batch_size, seq_len, num_v_heads, head_v_dim)

        # 调整 ba 
        beta = b.sigmoid()  # 转化为 0-1 之间的 β (batch_size, seq_len, num_v_heads)，它是完全输入相关的
        # 将 a 转换为 fp32，防止精度下溢，导致出现 -inf
        # Gated DeltaNet 中 α 参数化方法与 Mamba2 相同，即 α = exp(A·Δt)，通过参数化保证 A < 0 和 Δt > 0，从而使 α∈(0, 1)
        # 其中，A 代表每个 head 的基础时间尺度，Δt 代表时间步长，它包括 a 和 dt_bias
        #  - a 是输入经过 Linear 得到的，因此它是内容相关的动态调节量
        #  - dt_bias 是参数权重，训练后得到的是静态偏置
        #  - softplus(x) = log(1 + exp(x))，它的作用是将 Δt 控制为正
        # 因此这里 g = -exp(A_log) * softplus(a + dt_bias) = -A * softplus(a + dt_bias) 恒为负值
        # 随后 g 会通过 exp(g) 使其转化为 α∈(0, 1)，综合来看，α 由 A、a、dt_bias 共同决定
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # (batch_size, seq_len, num_v_heads)
        
        # 调整 qk 头数，类似于 softmax 注意力中的 repeat_kv
        # num_k_heads 起到寻址功能，而 num_v_heads 则具备内容功能
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # ------------------------- 4. 应用 Gated Delta Rule -------------------------
        # prefill 阶段，分 chunk 并行
        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        # decode 阶段，自回归
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        # core_attn_out: (batch_size, seq_len, num_heads, head_v_dim)
        # last_recurrent_state: (batch_size, num_heads, head_k_dim, head_v_dim)

        # 更新 cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # ------------------------- 5. 输出 -------------------------
        # 记录原始形状
        z_shape_og = z.shape  # (batch_size, seq_len, num_v_heads, head_v_dim)
        # 转换为 2D 张量
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])  # (batch_size * seq_len * num_v_heads, head_v_dim)
        z = z.reshape(-1, z.shape[-1])  # (batch_size * seq_len * num_v_heads, head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)  # RMSNorm + Gate
        core_attn_out = core_attn_out.reshape(z_shape_og)  # (batch_size, seq_len, num_v_heads, head_v_dim)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)  # (batch_size, seq_len, value_dim)

        output = self.out_proj(core_attn_out)  # (batch_size, seq_len, hidden_size)
        
        return output