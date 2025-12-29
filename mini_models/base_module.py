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

