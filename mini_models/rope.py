from typing import Tuple, Optional
import torch
from torch import nn
import math



# ----------------------------------------- RoPE 复数形式实现 -----------------------------------------
# RoPE 的复数形式实现，在一些源码中能够看到这种方式
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


# ----------------------------------------- RoPE 实数形式实现 -----------------------------------------
# 本项目采用 transformers 中的实数形式实现
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将输入张量 x 的最后一个维度分成两半, 交换位置并将前一半取反
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
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
        position_embeddings (Tuple[torch.Tensor, torch.Tensor]): 预计算的余弦表和正弦表元组 (cos, sin), 每个表的形状为 (batch, seq_len, head_dim)

    Returns:
        torch.Tensor: 应用了RoPE的输出张量
    """
    dtype = x.dtype
    cos, sin = position_embeddings  # (batch, seq_len, head_dim)

    # 增加 heads 维度
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


# 计算 YaRN 逆频率和注意力缩放参数
def compute_yarn_parameters(
    rope_theta: float, 
    rope_scaling: dict, 
    head_dim: int, 
    max_position_embeddings: int
) -> Tuple[torch.Tensor, float]:
    """
    计算 YaRN 扩展的逆频率和注意力缩放参数

    Args:
        rope_theta (float): RoPE 的底, 默认为 10000.0
        rope_scaling (dict): ROPE 的缩放参数, 包括以下字段:
            - rope_type (str): ROPE 扩展类型, 目前固定为 'yarn'
            - factor (float): 扩展倍数，即扩展后上下文长度/扩展前上下文长度，即论文中的 s
            - attention_factor (float, optional): 注意力缩放因子，即论文中的 √(1/t), 可以自定义, 默认为 None, 此时由 factor 计算得到
            - beta_fast (float, optional): 论文中的 β，用于划分高频部分，默认为 32
            - beta_slow (float, optional): 论文中的 α，用于划分低频部分，默认为 1
        head_dim (int): 每个头的维度
        max_position_embeddings (int): 扩展后的最大位置编码长度

    Returns:
        Tuple[torch.Tensor, float]: 逆频率张量 (head_dim//2,) 和注意力缩放因子
    """
    # 变量准备
    base = rope_theta
    dim = head_dim  # 默认全部维度使用 RoPE
    factor = rope_scaling["factor"]  # 扩展倍数，即扩展后上下文长度/扩展前上下文长度，即论文中的 s
    attention_factor = rope_scaling.get("attention_factor", None)  # 注意力缩放因子，即论文中的 √(1/t)，可以自定义，默认为 None，此时由 factor 计算得到
    beta_fast = rope_scaling.get("beta_fast")  # 论文中的 β，用于划分高频部分，默认为 32
    beta_slow = rope_scaling.get("beta_slow")  # 论文中的 α，用于划分低频部分，默认为 1
    original_max_position_embeddings = max_position_embeddings / factor  # 扩展前的最大位置编码长度

    # 计算注意力缩放因子
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0  # 论文中的 √(1/t)
    
    # (dim//2,)
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs  # 无需应用插值的逆频率
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)  # 应用插值的逆频率

    # ------------- 工具函数 -------------
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """
        根据给定的旋转次数, 计算对应的维度, 对应的公式是:
            r(d) = L / λ_d = L / [2πb^(2d/D)]
        经过简单的推导, 有:
            d = [D log(L / 2πr)] / [2 log(b)]
        在 D=64, b=10000, L=512, α=1, β=32 的配置下, β 对应的维度约为 3.25, α 对应维度约为 15.29
        
        Args:
            num_rotations (float): 旋转次数, 即 r(d) = L / λ_d, 表示原最大序列长度在此波长上走了几个周期
            dim (int): 总维度数, 即 D
            base (float): RoPE 的底, 即 b
            max_position_embeddings (int): 原最大位置编码长度, 即 L
        Returns:
            float: 对应的维度, 即 d
        """
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)  # α 对应的维度，高频低维度
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)  # β 对应的维度，低频高维度
        if truncate:
            low = math.floor(low)  # 向下取整
            high = math.ceil(high)  # 向上取整
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001  # 防止除零

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)  # (d - min) / (max - min)
        # torch.clamp(input, min, max) 用于将输入张量的每个元素限制在 [min, max] 区间内
        # 如果元素值 < min，就变成 min
        # 如果元素值 > max，就变成 max
        # 如果在 min 和 max 之间，就保持不变
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    truncate = True  # 默认取整
    # 根据给定的 α 和 β，找到维度分隔点
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    # NOTE: 在原论文中，有如下公式：
    #         ╭ 0,               if r < α
    # γ(r) =  | 1,               if r > β
    #         ╰ (r-α)/(β-α),     otherwise
    # 因此它在 otherwise 的情况下是对 r 的线性而言的
    # 而 transformers 的实现中，此时的 linear_ramp_factor 是通过将 r 解算出维度，在维度上的线性
    # < low 的低维度对应的范围是 r > β
    # > high 的高维度对应的范围是 r < α
    # 由于 r 与 d 不是线性映射，因此严格来说，原文是在 r 的尺度上均匀过渡，transformers 是在 d 的尺度上均匀过渡，二者是存在区别的
    # 这里通过 1 - linear_ramp_factor 来纠正应用插值的范围
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )

    return inv_freq, attention_factor


# RoPE 层，其前向用于预计算 cos、sin 表
class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)

    Args:
        max_position_embeddings (int): 最大位置编码长度
        head_dim (int): 每个头的维度
        rope_theta (float): RoPE 的底数, 默认为 10000.0
        rope_scaling (dict): ROPE 的缩放参数, 在经过 YaRN 训练后, 会固定到 config 里
    """

    inv_freq: torch.Tensor  # 用于类型标注(type hint)

    def __init__(self, max_position_embeddings: int, head_dim: int, rope_theta: float = 10000.0, rope_scaling: dict = None):
        super().__init__()

        # NOTE: 确定扩展类型，当前仅实现 YaRN 扩展，根据是否传入 rope_scaling 来确定是否使用 YaRN 扩展
        if rope_scaling:
            self.rope_type = "yarn"
        else:
            self.rope_type = "default"
        
        self.max_seq_len_cached = max_position_embeddings
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        if self.rope_type == "default":
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))  # (head_dim//2,)
            self.attention_scaling = 1.0  # 不对注意力进行缩放
        else:
            # 计算 YaRN 逆频率和注意力缩放参数
            inv_freq, self.attention_scaling = compute_yarn_parameters(rope_theta, rope_scaling, head_dim, max_position_embeddings)
        
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
            
            # 例如, 某个位置的角度为 [θ1, θ2], 则拼接的 angle 为[θ1, θ2, θ1, θ2]
            emb = torch.cat((freqs, freqs), dim=-1)  # (batch, seq_len, head_dim)
            
            # 直接在 cos sin 上应用缩放因子，可以在不影响注意力实现的情况下，实现对注意力的缩放
            cos = emb.cos() * self.attention_scaling # (batch, seq_len, head_dim) 该位置变为[cos(θ1), cos(θ2), cos(θ1), cos(θ2)]
            sin = emb.sin() * self.attention_scaling  # (batch, seq_len, head_dim) 该位置变为[sin(θ1), sin(θ2), sin(θ1), sin(θ2)]

        return cos.to(x.dtype), sin.to(x.dtype)
