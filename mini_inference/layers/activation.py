import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile  # torch.compile 优化，由于编译成本，有助于加速大型tensor的计算，对于小型tensor的计算反而会因为编译时间过长降低效率
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入张量在最后一维上是 gate 和 up projection 的拼接结果，chunk(2, -1) 把它拆成 x（gate 分支）和 y（up 分支）
        x, y = x.chunk(2, -1)
        return F.silu(x) * y  # 前一半做 SiLU，再乘以后一半
