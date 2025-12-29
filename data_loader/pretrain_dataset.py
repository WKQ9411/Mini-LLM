import numpy as np
import torch
from torch.utils.data import Dataset
import math
import os
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from .utils import print_aligned, format_size


class PreTrainDataset(Dataset):
    """
    使用 numpy.memmap 从大型二进制文件中读取数据，并使用带重叠的滑动窗口生成样本

    设总 tokens 长度为 m, 窗口长度为 w, 步长为 s:
    - 如果 (m-w) 能被 s 整除, 那么标准滑动窗口产生的最后一个样本就是序列的最后 w 个 tokens。总样本数就是标准样本数: (m-w) / s + 1
    - 如果 (m-w) 不能被 s 整除, 那么标准滑动窗口最后会余下来一些 tokens 未被采样, 如果是这种情况, 我们直接取 m 最后 w 个 tokens 作为最后一个样本, 总样本数是: floor((m-w) / s) + 2

    我们可以用向上取整函数 ceil 来统一这两种情况:
    - 如果 (m-w)/s 是整数 k (即 (m-w)%s == 0), 则样本数为 k+1, 此时 ceil((m-w)/s) = k
    - 如果 (m-w)/s 不是整数, 设其为 k.f (k 是整数部分, f 是小数部分), 则样本数为 floor(k.f) + 2 = k+2, 此时 ceil((m-w)/s) = k+1
    - 综上所述, 样本数可以用 ceil((m-w)/s) + 1 来表示
    - 当(m-w)/s 不是整数时, 需额外采样最后 w 个 tokens 作为最后一个样本, 其起始位置索引是 m - w

    备注:
    - floor(x) 表示向下取整, 返回 <= x 的最大整数
    - ceil(x) 表示向上取整, 返回 >= x 的最小整数

    Args:
        file_path (str): 包含 token id 的二进制文件的路径
        max_seq_len (int): 最大序列长度
        dtype (np.dtype): 二进制文件中 token 的 numpy 数据类型, 默认为 np.uint16
        overlap_ratio (float): 连续窗口之间重叠部分的比例, 默认为 0.1 必须小于 1.0.
    """

    def __init__(
        self,
        file_path: str,
        max_seq_len: int,
        dtype=np.uint16,
        overlap_ratio: float = 0.1,
    ):
        super().__init__()

        # 获取文件大小（字节）
        self.file_size_bytes = os.path.getsize(file_path)
        self.file_size = format_size(self.file_size_bytes)
        self.item_size_bytes = np.dtype(dtype).itemsize

        # 计算文件中的总 token 数
        self.total_tokens = self.file_size_bytes // self.item_size_bytes

        # 计算采样每个样本的窗口长度 (max_seq_len)
        self.max_seq_len = max_seq_len

        # 计算滑动窗口的步长，步长是 sample_len 的非重叠部分，math.floor 是向下取整
        self.stride = math.floor(self.max_seq_len * (1.0 - overlap_ratio))
        self.overlap = self.max_seq_len - self.stride

        # --- 计算样本数量和最后一个样本的起始位置 ---
        # m = self.total_tokens, w = self.sample_len, s = self.stride
        # 公式: N = ceil((m - w) / s) + 1
        numerator = self.total_tokens - self.max_seq_len
        self.num_samples = math.ceil(numerator / self.stride) + 1

        # 最后一个样本总是从 total_tokens - sample_len 开始
        self.last_sample_start = self.total_tokens - self.max_seq_len

        # 只在主进程打印数据集信息
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            info = {
                "file path": file_path,
                "file size": self.file_size,
                "token dtype": f"{dtype} ({self.item_size_bytes} bytes/token)",
                "total tokens": f"{self.total_tokens:,} (~ {self.total_tokens / 1_000_000_000:.2f} B tokens)",
                "window size": self.max_seq_len,
                "window overlap ratio": f"{overlap_ratio * 100:.1f} %",
                "window step": self.stride,
                "num samples": f"{self.num_samples:,}",
            }
            print("------------ pretrain dataset info ------------")
            print_aligned(info)
            print("-----------------------------------------------")

        # 以内存映射模式（只读）打开文件
        self.data = np.memmap(file_path, dtype=dtype, mode="r", shape=(self.total_tokens,))

    def __len__(self):
        """返回数据集中样本的总数"""
        return self.num_samples

    def __getitem__(self, idx):
        """
        获取索引为 idx 的样本

        Args:
            idx (int): 样本的索引 (0 到 num_samples - 1)

        Returns:
            (torch.Tensor, torch.Tensor): 返回 input 和 target, 格式为 torch.Tensor(dtype=torch.long)
        """
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"The index {idx} is out of range (total num samples: {self.num_samples})")

        # 确定样本的起始位置
        if idx == self.num_samples - 1:
            # 对于最后一个样本，使用预先计算好的起始位置，不论(m-w) 能否被 s 整除，都是一样的
            start_idx = self.last_sample_start
        else:
            # 对于其他样本，起始位置是 idx * stride
            start_idx = idx * self.stride

        # 从内存映射数组中提取样本
        end_idx = start_idx + self.max_seq_len
        input_ids = self.data[start_idx:end_idx]

        return torch.tensor(input_ids, dtype=torch.long)


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent
    tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))

    # --- 测试 PreTrainDataset ---
    pretrain_data_path = str(root_path / "data/pretrain_data/bin/deepctrl.bin")

    dataset = PreTrainDataset(
        file_path=pretrain_data_path,
        max_seq_len=512,
        dtype=np.uint16,
        overlap_ratio=0.1,
    )

    # 不打乱DataLoader，比较第一个样本最后部分和第二个样本开始部分文字，观察重叠情况
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for input_ids in loader:
        print(f"input_ids shape: {input_ids.shape}")
        print(f"1st sample:\n {tokenizer.decode(input_ids[0].tolist())}")
        print("-" * 30)
        print(f"2nd sample:\n {tokenizer.decode(input_ids[1].tolist())}")
        break
