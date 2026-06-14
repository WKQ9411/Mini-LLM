import torch
import torch.nn.functional as F
from torch import nn


class GeGLUFFN(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))
