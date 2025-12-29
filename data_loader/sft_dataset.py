import torch
from torch.utils.data import Dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from .utils import print_aligned, format_size
from tqdm import tqdm
import pandas as pd


class SFTDataset(Dataset):
    """
    SFT 数据集，从 parquet 文件读取数据

    注意: 
    1. 推荐使用 packing, 防止不同 batch 中有效 token 长度不同, 导致训练震荡(高有效 token 的 batch 相对于低有效 token 的 batch 梯度会被稀释),
    低有效 token 的梯度信号会更强, 导致模型倾向于输出短长度回复
    2. 会自动检测是否已经 packing (通过 sample_lengths 字段, 该字段只有经过 packing 的 parquet 文件才有)

    Args:
        file_path (str): sft 的 parquet 文件的路径
        max_seq_len (int): 最大序列长度
        ignore_index (int): 用于 padding 的 ignore index, 默认为 -100
    """
    def __init__(self, file_path: str, max_seq_len: int, ignore_index: int = -100):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.file_size_bytes = os.path.getsize(file_path)
        self.file_size = format_size(self.file_size_bytes)
        self.file_path = file_path
        
        # 从 parquet 文件读取
        print(f"Loading data from parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # 检查是否已经 packing（通过 sample_lengths 字段判断）
        is_packed = 'sample_lengths' in df.columns
        self.packing = is_packed
        
        if is_packed:
            print("Detected packed parquet file (with packing)")
            self.data, self.total_pad_token = self._load_from_packed_parquet(df)
        else:
            print("Detected non-packed parquet file (without packing)")
            self.data, self.total_pad_token = self._load_from_non_packed_parquet(df)

        info = {
            "file path": file_path,
            "file size": self.file_size,
            "packing": self.packing,
            "num samples": len(self.data),
            "pad ratio": f"{self.total_pad_token / (len(self.data) * self.max_seq_len):.2%}"
        }

        # 直接打印，默认 sft 使用单卡
        print("-------------- sft dataset info --------------")
        print_aligned(info)
        print("----------------------------------------------")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _load_from_non_packed_parquet(self, df: pd.DataFrame):
        """
        从非 packing 的 parquet 文件加载数据
        """
        total_pad_token = 0
        data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading from parquet"):
            token_ids = row['token_ids']
            labels = row['labels']
            position_ids = row['position_ids']
            length = row['length']
            attention_mask_1d = row.get('attention_mask', None)
            
            # 如果 parquet 中没有 attention_mask，则根据 length 构建
            if attention_mask_1d is None:
                attention_mask_1d = [1] * length + [0] * (self.max_seq_len - length)
            
            # 计算 padding token 数量
            pad_len = self.max_seq_len - length
            total_pad_token += pad_len
            
            data.append({
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "position_ids": torch.tensor(position_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_1d, dtype=torch.long)  # (seq_len,) transformers 内部会自动构建 causal mask
            })
        
        return data, total_pad_token

    def _load_from_packed_parquet(self, df: pd.DataFrame):
        """
        从已经 packing 的 parquet 文件加载数据，构建 2D 斜对角块 attention mask
        """
        total_pad_token = 0
        data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading packed parquet"):
            token_ids = row['token_ids']
            labels = row['labels']
            position_ids = row['position_ids']
            length = row['length']
            sample_lengths = row['sample_lengths']  # 每个样本的长度列表
            
            # 构建斜对角块 mask
            attention_mask = torch.full((self.max_seq_len, self.max_seq_len), torch.finfo(torch.float32).min)
            start = 0
            for sample_len in sample_lengths:
                end = start + sample_len
                # 样本内部 Causal Mask
                causal_mask = torch.tril(torch.ones((sample_len, sample_len), dtype=torch.bool))
                attention_mask[start:end, start:end] = torch.where(causal_mask, 0.0, torch.finfo(torch.float32).min)
                start = end
            
            # attention mask 方阵的最后几行，可能出现全为 pad 的情况，这可能会导致 softmax 出现 0/0 的情况
            # 因此这里把剩余的 padding 部分的斜对角线的元素设置为 0，softmax 时，全 pad 行的值就会变为 1
            if start < self.max_seq_len:
                # 将 padding 区域的对角线设为 0.0
                for i in range(start, self.max_seq_len):
                    attention_mask[i, i] = 0.0
            
            # 计算 padding token 数量
            pad_len = self.max_seq_len - length
            total_pad_token += pad_len
            
            data.append({
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "position_ids": torch.tensor(position_ids, dtype=torch.long),
                "attention_mask": attention_mask.unsqueeze(0)  # (1, seq_len, seq_len)
            })
        
        return data, total_pad_token


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    root_path = Path(__file__).parent.parent
    tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))

    # --- 测试 SFTDataset ---
    # step 1. 测试非 packing 的 parquet 文件
    sft_data_path = str(root_path / "data/sft_data/parquet/sampled_sft_data.parquet")
    
    print("Testing non-packed parquet file:")
    dataset = SFTDataset(
        file_path=sft_data_path,
        max_seq_len=512
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for idx, data in enumerate(loader):
        if idx == 0:  # 检查第一个样本
            print(f"input_ids shape: {data['input_ids'].shape}")
            print(f"labels shape: {data['labels'].shape}")
            print(f"position_ids shape: {data['position_ids'].shape}")
            print(f"attention_mask shape: {data['attention_mask'].shape}")
            
            # 打印 1st sample 和 2nd sample
            print(f"1st sample:\n {tokenizer.decode(data['input_ids'][0].tolist())}")
            print("-" * 30)
            print(f"2nd sample:\n {tokenizer.decode(data['input_ids'][1].tolist())}")
            break
    
    # step 2. 测试 packing 的 parquet 文件
    packed_sft_data_path = str(root_path / "data/sft_data/parquet/packed_sft_data.parquet")
    
    if os.path.exists(packed_sft_data_path):
        print("\nTesting packed parquet file:")
        dataset_packed = SFTDataset(
            file_path=packed_sft_data_path,
            max_seq_len=512
        )
        
        loader_packed = DataLoader(dataset_packed, batch_size=2, shuffle=False)
        for idx, data in enumerate(loader_packed):
            if idx == 0:  # 检查第一个样本
                print(f"input_ids shape: {data['input_ids'].shape}")
                print(f"labels shape: {data['labels'].shape}")
                print(f"position_ids shape: {data['position_ids'].shape}")
                print(f"attention_mask shape: {data['attention_mask'].shape}")

                # 可视化第一个数据的 attention mask
                plt.figure(figsize=(10, 8))
                mask_data = data['attention_mask'][0][0].numpy()
                im = plt.imshow(mask_data, cmap='viridis')
                plt.title("Attention Mask Visualization", fontsize=16)
                # 设置坐标轴标签
                plt.xlabel('Key Position', fontsize=12)
                plt.ylabel('Query Position', fontsize=12)
                # 移除网格线，保持清晰
                plt.grid(False)
                # 调整布局
                plt.tight_layout()
                plt.savefig(root_path / "assets/attention_mask_visualization.png")
                plt.close()

                print(f"1st sample:\n {tokenizer.decode(data['input_ids'][0].tolist())}")
                print("-" * 30)
                print(f"2nd sample:\n {tokenizer.decode(data['input_ids'][1].tolist())}")
                break