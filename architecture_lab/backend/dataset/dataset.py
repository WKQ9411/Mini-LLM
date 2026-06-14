import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    使用 numpy.memmap 从大型二进制文件中读取数据, 并使用带重叠的滑动窗口生成样本
    返回经过 shift 的 input_ids 和 labels

    设总 tokens 长度为 m, 窗口长度为 w (= max_seq_len + 1, 多出的 1 个 token 用于 shift), 步长为 s:
    - 样本数: ceil((m - w) / s) + 1
    - 最后一个样本总是从 m - w 开始

    Args:
        data_path (str): 包含 token id 的二进制文件的路径
        seq_len (int): 输出序列长度 (input_ids 和 labels 的长度)
        dtype (np.dtype): 二进制文件中 token 的 numpy 数据类型, 默认为 np.uint16
        overlap_ratio (float): 连续窗口之间重叠部分的比例, 默认为 0.1
    """

    def __init__(self, data_path: str, seq_len: int, dtype=np.uint16, overlap_ratio: float = 0.1):
        self.seq_len = seq_len
        # 实际需要从文件中读取的窗口长度，多出 1 个 token 用于构建 labels
        self.window_len = seq_len + 1

        file_size_bytes = os.path.getsize(data_path)
        item_size_bytes = np.dtype(dtype).itemsize
        total_tokens = file_size_bytes // item_size_bytes

        # 计算滑动窗口的步长（基于输出长度 seq_len，而非 window_len）
        self.stride = math.floor(seq_len * (1.0 - overlap_ratio))

        # 计算样本数量: ceil((m - w) / s) + 1
        numerator = total_tokens - self.window_len
        if numerator < 0:
            self.num_samples = 0
        else:
            self.num_samples = math.ceil(numerator / self.stride) + 1

        # 最后一个样本的起始位置
        self.last_sample_start = total_tokens - self.window_len

        self.data = np.memmap(data_path, dtype=dtype, mode="r")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"Index {idx} out of range (total samples: {self.num_samples})")

        # 确定起始位置
        if idx == self.num_samples - 1:
            start = self.last_sample_start
        else:
            start = idx * self.stride

        end = start + self.window_len
        chunk = torch.from_numpy(self.data[start:end].copy()).long()
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }
