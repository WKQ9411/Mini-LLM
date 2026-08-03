import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

# 保留 nano-vllm 中涉及到 TP 的 Linear 类
# LinearBase 是所有 Linear 的基类
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # nn.Linear(input_size, output_size)
        # weight shape: (output_size, input_size)
        #                ↑________↑  ↑_________↑
        #                第0维(行)    第1维(列)

        # 前向传播: output = input @ weight.T + bias
        #          (B, input_size) @ (input_size, output_size) → (B, output_size)
        self.tp_dim = tp_dim  # 用于指定在哪个维度进行tensor parallel，0表示输出维度（列并行），1表示输入维度（行并行）
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))  # weight shape: (output_size, input_size)
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 最简单的普通线性层
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.functional.linear(input, weight, bias) = input @ weight.T + bias
        return F.linear(x, self.weight, self.bias)


# NOTE: ColumnParallel（列并行）指的是对逻辑上的 y = xW 的 W 按列（输出维）切，这里的 W 是 (input_size, output_size)
# 但因为 weight 物理存成 (out, in)，输出维恰好是第 0 维（行），所以代码里 narrow(0, ...) 切的是"行"，这里容易混淆
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data  # param 已经位于 GPU，loaded_weights 是原始的完整权重，在 CPU 上
        shard_size = param_data.size(self.tp_dim)  # 获取当前 shard 的大小
        start_idx = self.tp_rank * shard_size  # 本地 GPU 的 shard 在原始权重中的起始索引
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)  # narrow(dim, start, length) 返回沿指定维度的切片
        param_data.copy_(loaded_weight)  # CPU -> GPU 的数据拷贝

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# 在 ColumnParallelLinear 基础上, 把若干个输出维相等的列并行矩阵合并成一个 param
# 典型用例是 FFN 的 gate_proj + up_proj 合并成 gate_up: 一次 matmul 同时算出 gate 和 up
# 再由 SiluAndMul 做 silu(gate)*up, 省掉一次 kernel launch 与一次权重访存
# NOTE: 这里“合并 + 按输出维 //tp_size 切分”的前提是各子矩阵本身没有内部结构约束
# (gate/up 只是普通线性层, 无 head 概念), 因此直接按输出维均分即可
# QKV 合并涉及 head 边界与 GQA 头数不等, 不应使用本类, 而应由 QKVColumnParallelLinear 处理
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],  # 各子矩阵的完整输出维, 例如 gate_up: [intermediate, intermediate]
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # param: 分片后待加载的本地参数 (在 GPU 上)
        # loaded_weights: 原始完整 checkpoint 权重 (在 CPU 上, 未分片)
        # loaded_weight_id: 当前加载的是第几个子矩阵 (例如 gate_up 中 0 -> gate, 1 -> up)
        
        # 以 gate_up 为例 (tp_size=2, intermediate=8, input_size=4):
        # checkpoint = {
        #     'mlp.gate_proj.weight': (8, 4),
        #     'mlp.up_proj.weight':   (8, 4),
        # }
        # 合并后的本地 param 形状为 (sum([8,8])//2, 4) = (8, 4), 行布局 [gate_shard(4) | up_shard(4)]
        # - 加载 gate (id=0): offset=0, shard_size=4, 落到 param[0:4)
        # - 加载 up  (id=1): offset=8//2=4, shard_size=4, 落到 param[4:8)
        # 每个 rank 再从完整 checkpoint 中切出本卡那片 copy 进对应子区间
        
        param_data = param.data  # 合并后的本地 param, 形状 (sum(output_sizes)//tp_size, input_size), 位于 GPU, 行布局为各子矩阵分片顺序拼接
        # 在已分片的 param 中定位当前子矩阵的子区间起点，当 loaded_shard_id 为 0 时，sum([]) 为 0，表示从 param 的开头开始加载
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size  # 当前子矩阵在本卡上的行数
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)  # 取出合并 param 中当前子矩阵对应的子区间
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]  # 将原始权重按 tp_size 均分, 取出当前 rank 的分片
        param_data.copy_(loaded_weight)


# 涉及到 head 的 QKV 合并
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)  # 每个 GPU 上的 num_heads
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # 每个 GPU 上的 num_kv_heads
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size  # 总输出维度 = q + k + v
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # load_weight_id: q, k, v
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# NOTE: RowParallel（行并行）指的是对逻辑上的 y = xW 的 W 按行（输入维）切，这里的 W 是 (input_size, output_size)
# 但因为 weight 物理存成 (out, in)，输入维恰好是第 1 维（列），所以代码里 narrow(1, ...) 切的是"列"
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:  # 行并行是输入维度切分，此时是 bias，处于输出空间，每张卡都一样，直接拷贝即可
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只在第 0 rank 上有 bias，其他 rank 上的 bias 为 None
        # 否则 bias 会被重复加到每个 rank 的输出上，导致多加
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


# 在不使用 TP 的情况下的合并投影
class MergedLinear(nn.Linear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        super().__init__(input_size, sum(output_sizes), bias=bias)
        self.output_sizes = output_sizes
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        if not 0 <= loaded_shard_id < len(self.output_sizes):
            raise ValueError(f"Invalid shard id: {loaded_shard_id}")
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        param_shard = param.data.narrow(0, shard_offset, shard_size)
        if param_shard.shape != loaded_weight.shape:
            raise ValueError(
                f"Cannot load shape {tuple(loaded_weight.shape)} into merged shard "
                f"with shape {tuple(param_shard.shape)}"
            )
        param_shard.copy_(loaded_weight)
