from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------------------- Norm -----------------------------------------
class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization, LayerNorm)

    Args:
        dim (int): 嵌入维度
        eps (float): Epsilon 值用于确保数值稳定性, 默认为 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        """
        LayerNorm 前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 归一化后的输出
        """
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


class RMSNorm(nn.Module):
    """
    均方根归一化 (Root Mean Square Layer Normalization, RMSNorm)

    Args:
        dim (int): 嵌入维度
        eps (float): Epsilon 值用于确保数值稳定性, 默认为 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm 前向传播, 在dim维度上进行归一化

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 归一化后的输出
        """
        return F.rms_norm(x.float(), (self.dim,), self.weight, self.eps).type_as(x)


class ZeroCenteredRMSNorm(nn.Module):
    """
    零中心均方根归一化 (Zero-Centered Root Mean Square Layer Normalization, Zero-Centered RMSNorm)
    与 RMSNorm 不同的是, 初始化参数为 0, 缩放系数为 (1 + weight)

    Args:
        dim (int): 嵌入维度
        eps (float): Epsilon 值用于确保数值稳定性, 默认为 1e-6
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # 初始化参数为 0

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zero-Centered RMSNorm 前向传播, 在dim维度上进行归一化

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 归一化后的输出
        """
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class RMSNormGated(nn.Module):
    """
    带门控机制的均方根归一化 (RMSNorm with Gating)
    
    Args:
        hidden_size (int): 嵌入维度
        eps (float): Epsilon 值用于确保数值稳定性, 默认为 1e-6
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        RMSNormGated 前向传播, 在dim维度上进行归一化

        Args:
            hidden_states (torch.Tensor): 输入张量
            gate (torch.Tensor): 门控张量

        Returns:
            torch.Tensor: 归一化后的输出
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        
        # Norm
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        
        # Gate
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))  # 使用 SiLU 激活函数作为门控机制

        return hidden_states.to(input_dtype)


# ----------------------------------------- FeedForward -----------------------------------------
class SwiGLUFFN(nn.Module):
    """
    带 SwiGLU 激活函数的前馈层

    Args:
        dim (int): 嵌入维度
        inter_dim (int): 隐藏层维度
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU_FFN 前向传播

        Args:
            x (torch.Tensor): 输入张量 (batch_size, seq_len, dim)

        Returns:
            torch.Tensor: 输出张量 (batch_size, seq_len, dim)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

