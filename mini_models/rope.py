from typing import Tuple, Optional

import torch
from torch import nn


# ----------------------------------------- Position Embedding -----------------------------------------
# 这里同时给出复数实现形式和实数实现形式，但实数实现形式更兼容，本项目仅采用实数实现形式
# 1. 复数形式实现
def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 复数频率矩阵, 并将其表示为复数的极坐标表示, 函数名中的 cis 指 cos(θ)+i·sin(θ), 表示一个复数位于单位圆上的位置

    Args:
        head_dim (int): 每个头的维度
        seq_len (int): 最大序列长度
        theta (float): RoPE 的底, 默认为10000.0

    Returns:
        torch.Tensor: 预计算的复数位置编码矩阵 (seq_len, head_dim//2)
    """
    # 计算不同维度的频率
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    # 计算位置索引
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)

    # 转换为复数形式
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb_complex(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    在复数域应用旋转位置编码, 将输入张量与复数频率矩阵相乘, 得到应用了 RoPE 的输出张量

    Args:
        x (torch.Tensor): 输入张量 (batch, heads, seq_len, head_dim)
        freqs_cis (torch.Tensor): 预计算的复数位置编码矩阵 (seq_len, head_dim//2), 需根据输入 x 的形状和位置切片好

    Returns:
        torch.Tensor: 应用了RoPE的输出张量
    """
    dtype = x.dtype
    # 将 head_dim 维度进行变换并转换为复数
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))  # (batch, heads, seq_len, head_dim//2)
    freqs_cis = freqs_cis.view(1, 1, x.size(2), x.size(-1))  # (1, 1, seq_len, head_dim//2)
    y = torch.view_as_real(x * freqs_cis).flatten(3)  # (batch, heads, seq_len, head_dim)
    return y.to(dtype)


# 2. 实数形式实现
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将输入张量 x 的最后一个维度分成两半, 交换位置并将前一半取反
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_cos_sin_tables(head_dim: int, seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 余弦表和正弦表

    Args:
        head_dim (int): 每个头的维度
        seq_len (int): 最大序列长度
        theta (float): RoPE 的底, 默认为10000.0

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 预计算的余弦表和正弦表元组 (cos, sin), 每个表的形状为 (seq_len, head_dim)
    """
    # 计算不同维度的频率
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    # 计算位置索引
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # 每个位置每个维度的旋转角度 (seq_len, head_dim//2)
    angle = torch.cat([freqs, freqs], dim=-1)  # 例如, 某个位置的角度为 [θ1, θ2], 则拼接的 angle 为[θ1, θ2, θ1, θ2], 形状为 (seq_len, dim)

    # 计算余弦表和正弦表
    cos = torch.cos(angle)  # (seq_len, head_dim) 该位置变为[cos(θ1), cos(θ2), cos(θ1), cos(θ2)]
    sin = torch.sin(angle)  # (seq_len, head_dim) 该位置变为[sin(θ1), sin(θ2), sin(θ1), sin(θ2)]
    freaqs_cos_sin = (cos, sin)
    return freaqs_cos_sin


def apply_rotary_emb_real(x: torch.Tensor, freqs_cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    在实数域应用旋转位置编码, 基本原理如下:

    对于每一对维度 (a, b)，旋转角度 θ 后的新向量 (a', b') 是:

        ⎡ a'⎤ = ⎡ cos(θ)  -sin(θ)⎤ ⎡ a ⎤
        ⎣ b'⎦   ⎣ sin(θ)  cos(θ) ⎦ ⎣ b ⎦

    展开得:

        a' = a * cos(θ) - b * sin(θ)
        b' = b * cos(θ) + a * sin(θ)

    假设:
        x = [a, b]
        cos = [cosθ, cosθ]
        sin = [sinθ, sinθ]
    则:
        x * cos = [a cosθ, b cosθ]
        rotate_half(x) = [-b, a]
        rotate_half(x) * sin = [-b sinθ, a sinθ]
    相加得:
        [a' b'] = [a cosθ - b sinθ, b cosθ + a sinθ] = x * cos + rotate_half(x) * sin

    Args:
        x (torch.Tensor): 输入张量 (batch, heads, seq_len, head_dim)
        freqs_cos_sin (Tuple[torch.Tensor, torch.Tensor]): 预计算的余弦表和正弦表元组 (cos, sin), 每个表的形状为 (batch, seq_len, dim), 需根据输入 x 的形状和位置切片好

    Returns:
        torch.Tensor: 应用了RoPE的输出张量
    """
    dtype = x.dtype
    cos, sin = freqs_cos_sin  # (batch, seq_len, head_dim)

    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)

    # 假设 x 有四个维度 [a, b, c, d]，则 rotate_half(x) 后变为 [-c, -d, a, b]
    # 对应的旋转角度 angle 为 [θ1, θ2, θ1, θ2]，因此，实际上 a 与 c 组合为一对, b 与 d 组合为一对，然后进行旋转
    # 这与复数形式相邻维度组成一对旋转不同，不过理论上效果是一样的
    # 计算过程为:
    # x*cos = [a*cos(θ1), b*cos(θ2), c*cos(θ1), d*cos(θ2)]
    # rotate_half(x)*sin = [-c*sin(θ1), -d*sin(θ2), a*sin(θ1), b*sin(θ2)]
    # 相加得：[a*cos(θ1) - c*sin(θ1), b*cos(θ2) - d*sin(θ2), a*cos(θ1) + c*sin(θ1), b*cos(θ2) + d*sin(θ2)] -> [a', b', c', d']
    return ((x.float() * cos) + (rotate_half(x).float() * sin)).to(dtype)


# 3. 统一接口实现
def apply_rotary_emb(
    x: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    impl: str = "real",
) -> torch.Tensor:
    """
    应用旋转位置编码, 可选复数实现方式或实数实现方式
    复数实现方式更加直观, 但复数数据类型可能兼容有限, 复数实现中使用的是相邻维度为一组
    实数实现方式更通用, 但不那么直观, 实数实现中使用的是拆分后, 对应的维度为一组

    Args:
        x (torch.Tensor): 输入张量 (batch, heads, seq_len, head_dim)
        position_embeddings (Tuple[torch.Tensor, torch.Tensor] | torch.Tensor): 预计算的余弦表和正弦表元组 (cos, sin), 每个表的形状为 (seq_len, dim), 需根据输入 x 的形状和位置切片好
        impl (str): 实现方式, 默认为 "real", 可选 "real" 和 "complex"

    Returns:
        torch.Tensor: 应用了 RoPE 的输出张量
    """
    if impl == "real":
        return apply_rotary_emb_real(x, position_embeddings)
    elif impl == "complex":
        return apply_rotary_emb_complex(x, position_embeddings)
    else:
        raise ValueError(f"Invalid implementation: {impl}, must be 'real' or 'complex'")


def precompute_position_embeddings(
    max_position_embeddings: int,
    head_dim: int,
    theta: float = 10000.0,
    impl: str = "real",
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    预计算位置编码嵌入, 可选复数实现方式或实数实现方式

    Args:
        max_position_embeddings (int): 最大位置编码长度
        head_dim (int): 每个头的维度
        theta (float): RoPE 的底, 默认为 10000.0
        impl (str): 实现方式, 默认为 "real", 可选 "real" 和 "complex"

    Returns:
        Tuple[torch.Tensor, torch.Tensor] | torch.Tensor: 预计算的余弦表和正弦表元组或预计算的复数位置编码矩阵
    """
    if impl == "real":
        return precompute_cos_sin_tables(head_dim, max_position_embeddings, theta)
    elif impl == "complex":
        return precompute_freqs_cis(head_dim, max_position_embeddings, theta)
    else:
        raise ValueError(f"Invalid implementation: {impl}, must be 'real' or 'complex'")


# 类 transformers 的实现形式
class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)

    Args:
        max_position_embeddings (int): 最大位置编码长度
        head_dim (int): 每个头的维度
        rope_theta (float): RoPE 的底数, 默认为 10000.0
    """

    inv_freq: torch.Tensor  # 用于类型标注(type hint)

    def __init__(self, max_position_embeddings: int, head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))  # (head_dim//2,)
        # 仅缓存 inv_freq，而不是 cos、sin，能够节省缓存，且支持动态适应
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行前向传播会计算产生 cos、sin 表, 可以自适应序列长度, 可以处理训练时未见过的序列长度

        Args:
            x (torch.Tensor): 输入的 embeddings, 形状为 (batch, seq_len, hidden_size)
            position_ids (torch.Tensor): 位置索引, 形状为 (batch, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输出 cos、sin 表, 形状为 (batch, seq_len, head_dim)
        """
        # 调整形状为后续外积计算做准备
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)  # (batch, head_dim//2, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # (batch, 1, seq_len)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        # 关闭自动混合精度，强制使用 float32 计算，确保 cos、sin 表的精度
        with torch.autocast(device_type=device_type, enabled=False):
            # 批量矩阵乘法，freqs 是每个 token 在每个频率维度上的旋转角度
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # (batch, seq_len, head_dim//2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (batch, seq_len, head_dim) 详见 precompute_cos_sin_tables 函数中的注释
            cos = emb.cos()  # (batch, seq_len, head_dim)
            sin = emb.sin()  # (batch, seq_len, head_dim)

        return cos.to(x.dtype), sin.to(x.dtype)
