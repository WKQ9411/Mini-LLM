import numpy as np
import torch
from torch.utils.data import Dataset
import math
import os
from utils.little_tools import print_aligned
import torch.distributed as dist
from transformers import AutoTokenizer
import pandas as pd


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
        max_seq_len (int): 最大序列长度, 每个样本的长度将是 max_seq_len + 1
        dtype (np.dtype): 二进制文件中 token 的 numpy 数据类型, 默认为 np.uint16
        overlap_ratio (float): 连续窗口之间重叠部分的比例 (相对于 max_seq_len + 1), 默认为 0.1 必须小于 1.0.
    """
    def __init__(self, file_path: str, max_seq_len: int, dtype=np.uint16, overlap_ratio: float = 0.1):
        super().__init__()

        # 获取文件大小（字节）
        self.file_size_bytes = os.path.getsize(file_path)
        self.item_size_bytes = np.dtype(dtype).itemsize

        # 计算文件中的总 token 数
        self.total_tokens = self.file_size_bytes // self.item_size_bytes

        # 计算采样每个样本的窗口长度 (max_seq_len + 1)
        self.sample_len = max_seq_len + 1

        # 计算滑动窗口的步长，步长是 sample_len 的非重叠部分，math.floor 是向下取整
        self.stride = math.floor(self.sample_len * (1.0 - overlap_ratio))
        self.overlap = self.sample_len - self.stride

        # --- 计算样本数量和最后一个样本的起始位置 ---
        # m = self.total_tokens, w = self.sample_len, s = self.stride
        # 公式: N = ceil((m - w) / s) + 1
        numerator = self.total_tokens - self.sample_len
        self.num_samples = math.ceil(numerator / self.stride) + 1

        # 最后一个样本总是从 total_tokens - sample_len 开始
        self.last_sample_start = self.total_tokens - self.sample_len

        # 只在主进程打印数据集信息
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            info = {
                "文件路径": file_path,
                "文件大小": f"{self.file_size_bytes / (1024*1024*1024):.2f} GB",
                "Token 数据类型": f"{dtype} (每个 token {self.item_size_bytes} 字节)",
                "Token 总数": f"{self.total_tokens:,} (~ {self.total_tokens / 1_000_000_000:.2f} B Tokens)",
                "窗口长度": self.sample_len,
                "窗口重叠率": f"{overlap_ratio*100:.1f} %",
                "滑动步长": self.stride,
                "样本数量": f"{self.num_samples:,}",
            }
            print("------------ 数据集信息 ------------")
            print_aligned(info)
            print("-----------------------------------")

        # 以内存映射模式（只读）打开文件
        self.data = np.memmap(file_path, dtype=dtype, mode='r', shape=(self.total_tokens,))

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
            raise IndexError(f"索引 {idx} 超出范围 (数据集大小为 {self.num_samples})")
        
        # 确定样本的起始位置
        if idx == self.num_samples - 1:
            # 对于最后一个样本，使用预先计算好的起始位置，不论(m-w) 能否被 s 整除，都是一样的
            start_idx = self.last_sample_start
        else:
            # 对于其他样本，起始位置是 idx * stride
            start_idx = idx * self.stride

        # 从内存映射数组中提取样本
        end_idx = start_idx + self.sample_len
        sample = self.data[start_idx : end_idx]

        input_id = sample[:-1]
        target_id = sample[1:]

        return torch.tensor(input_id, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)


class SFTDataset(Dataset):
    """
    SFT 数据集

    备注: 【tokenizer 在 SFTDataset 类内使用, 如果更改了 tokenizer, 可能需要更改 __getitem__ 中分词的逻辑】

    Args:
        file_path (str): SFT 的 csv文件的路径
        max_seq_len (int): 最大序列长度, 每个样本的长度将是 max_seq_len + 1
        tokenizer (AutoTokenizer): 用于将文本转换为 token 的 tokenizer
    """
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_seq_len: int):
        super().__init__()

        self.df = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.sample_len = max_seq_len + 1

        file_size_bytes = os.path.getsize(file_path)
        def format_size(size_bytes):
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.2f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/1024**2:.2f} MB"
            else:
                return f"{size_bytes/1024**3:.2f} GB"

        file_size = format_size(file_size_bytes)

        info = {
            "文件路径": file_path,
            "文件大小": f"{file_size}",
            "样本数量": f"{len(self.df)}",
        }

        # 直接打印，默认 sft 使用单卡
        print("------------ 数据集信息 ------------")
        print_aligned(info)
        print("-----------------------------------")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        # 不考虑带有 history 的微调
        q = sample['q']
        a = sample['a']
        
        messages = [
            {"role": "user", "content": f'{q}'},
            {"role": "assistant", "content": f"{a}"}
        ]
        
        sample_ids = self.tokenizer.apply_chat_template(messages, tokenize=True)
        sample_ids = sample_ids[:self.sample_len]  # 如果大于最大长度，就截断，max_seq_len + 1
        pad_len = self.sample_len - len(sample_ids)  # 填充<pad>的长度
        sample_ids += [self.tokenizer.unk_token_id] * pad_len  # TODO: 这里使用<unk>补全，后续可新增<pad>
        input_id = torch.tensor(sample_ids[:-1], dtype=torch.long)  # 构造input，max_seq_len
        
        # 将不是需要模型生成的部分的 id 替换为 -100, -100是 nn.CrossEntropyLoss 默认的需要忽略计算 loss 的标签
        # 需要替换的有 <s>system 到 <s>assistant 之间的部分和 <pad> 部分
        prompt_messages = messages[:-1] # 只包含 System 和 User
        prompt_ids = self.tokenizer.apply_chat_template(prompt_messages, tokenize=True)
        prompt_len = len(prompt_ids)
        # 将 sample_ids 的前 prompt_len 个和后 pad_len 个 id 替换为 -100
        sample_ids[:prompt_len] = [-100] * prompt_len
        if pad_len > 0:
            sample_ids[-pad_len:] = [-100] * pad_len
        target_id = torch.tensor(sample_ids[1:], dtype=torch.long)  # 构造target，max_seq_len
        
        return input_id, target_id


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained("./mini_tokenizer")
    
    # --- 测试 PreTrainDataset ---
    # pretrain_data_path = './preprocess_data/data/pretrain_data/pretrain_data.bin'

    # dataset = PreTrainDataset(
    #     file_path=pretrain_data_path,
    #     max_seq_len=512,
    #     dtype=np.uint16,
    #     overlap_ratio=0.1
    # )

    # # 不打乱DataLoader，比较第一个样本最后部分和第二个样本开始部分文字，观察重叠情况
    # loader = DataLoader(dataset, batch_size=2, shuffle=False)
    # for input, target in loader:
    #     print(f"input shape: {input.shape}, target shape: {target.shape}")
    #     print(f"第一条数据内容为:\n {tokenizer.decode(input[0].tolist())}")
    #     print('-'*30)
    #     print(f"第二条数据内容为:\n {tokenizer.decode(input[1].tolist())}")
    #     break

    # --- 测试 SFTDataset ---
    # sft_data_path = './preprocess_data/data/sft_data/sft_data.csv'
    sft_data_path = "F:/BaiduNetdiskDownload/data/sft_data/sft_data_zh.csv"

    dataset = SFTDataset(
        file_path=sft_data_path,
        tokenizer=tokenizer,
        max_seq_len=512,
    )
 
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    total_step = len(loader)
    for input, target in tqdm(loader, total=total_step, desc="checking data shape"):
        # print(f"input shape: {input.shape}, target shape: {target.shape}")
        # print(f"第一条数据内容为:\n {tokenizer.decode(input[0].tolist())}")
        # print('-'*30)
        # print(f"第二条数据内容为:\n {tokenizer.decode(input[1].tolist())}")
        continue
        