from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import json
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
from collections import Counter


# 定义路径
root_path = Path(__file__).parent.parent
# pretrain
pretrain_data_path = root_path / "data/pretrain_data"
processed_pretrain_data_path = pretrain_data_path / "bin"
processed_pretrain_data_path.mkdir(parents=True, exist_ok=True)
# sft
sft_data_path = root_path / "data/sft_data"

# 加载训练好的分词器路径
tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))
eos_token = tokenizer.eos_token
pad_token = tokenizer.pad_token


# ----------------------------------------- 数据集路径 -----------------------------------------
# OpenCSG Fineweb-Edu-Chinese-V2.1 数据集 - pretrain / tokenizer
# https://www.modelscope.cn/datasets/opencsg/Fineweb-Edu-Chinese-V2.1
fineweb_edu_file_path = pretrain_data_path / "fineweb_edu"
fineweb_edu_bin_path = processed_pretrain_data_path / "fineweb_edu.bin"

# deepctrl 数据集 - pretrain /sft
# https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data
deepctrl_file_path = sft_data_path / "deepctrl/sft_data_zh.jsonl"
deepctrl_bin_path = processed_pretrain_data_path / "deepctrl.bin"

# ------------------------------------- pretrain data 构造 -------------------------------------
# 在 Mini-LLM V1 中，预训练数据集的构造方式是: <bos> text <eos>
# 在 Mini-LLM V2 中，我们省略 <bos> ，构造为: text <eos>

# 处理 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集
def process_pretrain_fineweb_edu(data_path: str, bin_path: str, buffer_size: int = 1000000, dtype_str: str = 'uint16'):
    """
    读取 parquet 文件目录, 提取文本字段, 分词, 并将 token id 保存为二进制文件

    Args:
        data_path (str): 输入的 parquet 文件路径
        bin_path (str): 输出的二进制文件路径
        buffer_size (int): 写入磁盘前在内存中缓冲的 token id 数量
        dtype_str (str): 保存 token id 的 numpy 数据类型 ('uint16', 'uint32'等), 'uint16' 适用于词汇量 < 65536 的分词器, 如果词汇量更大，请使用 'uint32'
    """
    # 选择合适的 NumPy 数据类型
    try:
        dtype = np.dtype(dtype_str)
    except TypeError:
        raise TypeError(f"Invalid dtype_str '{dtype_str}'. Please use 'uint16' or 'uint32'.")

    print(f"vocab size: {len(tokenizer)}")
    if dtype == np.uint16 and len(tokenizer) > 65535:
        raise ValueError(f"The vocabulary size of your tokenizer ({len(tokenizer)}) is too large for 'uint16'. Please use 'uint32' instead and try again.")
    
    token_buffer = []
    total_tokens = 0

    print(f"Start processing: {data_path}")
    
    # 获取所有parquet文件
    parquet_files = list(Path(data_path).glob("**/*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {data_path}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")

    with open(bin_path, 'wb') as f_out:
        # 遍历所有parquet文件
        for parquet_file in tqdm(parquet_files, desc="Processing files"):
            
            # 读取parquet文件
            df = pd.read_parquet(parquet_file)
            total_rows = len(df)

            # 使用 tqdm 显示进度
            for _, row in tqdm(df.iterrows(), total=total_rows, desc="Processing rows", leave=False):
                # 提取文本
                text = row.get('text')
                if text is None or pd.isna(text):
                    continue
                
                # 分词
                token_ids = tokenizer.encode(text + eos_token)
                # 添加到缓冲区
                token_buffer.extend(token_ids)
                
                # 如果缓冲区达到大小，则写入文件
                if len(token_buffer) >= buffer_size:
                    array_to_write = np.array(token_buffer[:buffer_size], dtype=dtype)
                    array_to_write.tofile(f_out)
                    total_tokens += len(array_to_write)
                    token_buffer = token_buffer[buffer_size:]  # 保留剩余部分

        # 处理结束后，写入缓冲区中剩余的 token
        if token_buffer:
            array_to_write = np.array(token_buffer, dtype=dtype)
            array_to_write.tofile(f_out)
            total_tokens += len(array_to_write)

    print("-" * 30)
    print(f"Processing completed!")
    print(f"Totally write {total_tokens:,} token ids to: {bin_path}")
    print(f"The data type is: {dtype.name}")
    print("-" * 30)


# 处理 deepctrl 数据集
def process_pretrain_deepctrl(data_path: str, bin_path: str, buffer_size: int = 1000000, dtype_str: str = 'uint16'):
    """
    读取 jsonl 文件, 提取文本字段, 分词, 并将 token id 保存为二进制文件

    Args:
        data_path (str): 输入的 jsonl 文件路径
        bin_path (str): 输出的二进制文件路径
        buffer_size (int): 写入磁盘前在内存中缓冲的 token id 数量
        dtype_str (str): 保存 token id 的 numpy 数据类型 ('uint16', 'uint32'等), 'uint16' 适用于词汇量 < 65535 的分词器, 如果词汇量更大，请使用 'uint32'
    """
    # 选择合适的 NumPy 数据类型
    try:
        dtype = np.dtype(dtype_str)
    except TypeError:
        raise TypeError(f"Invalid dtype_str '{dtype_str}'. Please use 'uint16' or 'uint32'.")

    print(f"vocab size: {len(tokenizer)}")
    if dtype == np.uint16 and len(tokenizer) > 65535:
        raise ValueError(f"The vocabulary size of your tokenizer ({len(tokenizer)}) is too large for 'uint16'. Please use 'uint32' instead and try again.")
    
    token_buffer = []
    total_tokens = 0

    print(f"Start processing: {data_path}")
    
    # 先计算总行数
    print("Calculating total lines of the file. This may take a while for large files ...")
    with open(data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        print(f"Total lines: {total_lines:,}")
    
    # 读取JSONL文件
    with open(data_path, 'r', encoding='utf-8') as f_in, open(bin_path, 'wb') as f_out:
        for line in tqdm(f_in, total=total_lines, desc="Processing lines"):
            data = json.loads(line.strip())
            
            # 构造文本内容，将instruction、input、output字段文本直接拼接
            text_parts = []
            
            # 添加instruction字段
            if data.get('instruction') and data['instruction'].strip():
                text_parts.append(data['instruction'].strip())
            # 添加input字段
            if data.get('input') and data['input'].strip():
                text_parts.append(data['input'].strip())
            # 添加output字段
            if data.get('output') and data['output'].strip():
                text_parts.append(data['output'].strip())
            # 如果没有任何有效文本，跳过
            if not text_parts:
                continue
            
            # 拼接所有文本部分
            combined_text = ''.join(text_parts)
            # 分词
            token_ids = tokenizer.encode(combined_text + eos_token)
            
            # 添加到缓冲区
            token_buffer.extend(token_ids)
            
            # 如果缓冲区达到大小，则写入文件
            if len(token_buffer) >= buffer_size:
                array_to_write = np.array(token_buffer[:buffer_size], dtype=dtype)
                array_to_write.tofile(f_out)
                total_tokens += len(array_to_write)
                token_buffer = token_buffer[buffer_size:]  # 保留剩余部分
        
        # 处理结束后，写入缓冲区中剩余的 token
        if token_buffer:
            array_to_write = np.array(token_buffer, dtype=dtype)
            array_to_write.tofile(f_out)
            total_tokens += len(array_to_write)

    print("-" * 30)
    print(f"Processing completed!")
    print(f"Totally write {total_tokens:,} token ids to: {bin_path}")
    print(f"The data type is: {dtype.name}")
    print("-" * 30)


# 按比例分层抽样 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集
def analyze_and_sample_fineweb_edu(data_path: str, output_path: str, sample_ratio: float = 0.1, max_rows_per_file: int = 1000000):
    """
    分析 fineweb_edu 数据集的 source 分布, 并按比例进行分层随机抽样
    
    Args:
        data_path (str): 原数据的 parquet 文件目录路径
        output_path (str): 输出抽样数据的目录路径
        sample_ratio (float): 抽样比例, 默认为 0.1(10%)
        max_rows_per_file (int): 保存采样文件时, 会检查是否超过此长度, 一旦超过就会保存为一个 parquet 文件, 默认为 1000000
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 获取所有 parquet 文件
    parquet_files = list(Path(data_path).glob("**/*.parquet"))
    if not parquet_files:
        print(f"Directory `{data_path}` does not contain any parquet files")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
        
    # 设置assets路径，用于保存原数据的 source 分布饼状图
    assets_path = Path(__file__).parent.parent / "assets"
    assets_path.mkdir(parents=True, exist_ok=True)
    
    # 检查是否存在已计算的source_counts文件
    source_counts_file = Path(data_path) / "source_counts.json"
    
    # ------------------ 1. 分析原数据的 source 分布 ------------------
    if source_counts_file.exists():
        print("Found existing source counts file, loading...")
        with open(source_counts_file, 'r', encoding='utf-8') as f:
            source_counts = json.load(f)
        source_counts = Counter(source_counts)
    else:
        print("No existing source counts file found, analyzing original data...")
        
        # 统计原数据的 source 分布
        source_counts = Counter()
        total_rows = 0
        
        print("Counting source distribution...")
        for parquet_file in tqdm(parquet_files, desc="Analyzing original data"):
            df = pd.read_parquet(parquet_file)
            total_rows += len(df)
            
            # 统计source分布
            if 'source' in df.columns:
                source_counts.update(df['source'].value_counts().to_dict())
        
        print(f"Total rows: {total_rows}")
        
        # 打印source分布统计
        print("\nSource distribution of original data:")
        total_count = sum(source_counts.values())
        for source, count in source_counts.most_common():
            percentage = (count / total_count) * 100
            print(f"    {source}: {count} ({percentage:.2f}%)")
        
        # 绘制原数据 source 分布饼状图并保存到assets文件夹
        plt.figure(figsize=(10, 8))
        sources = list(source_counts.keys())
        counts = list(source_counts.values())
        
        # 创建饼状图
        plt.pie(counts, labels=sources, autopct='%1.1f%%', startangle=90)
        plt.title('Source distribution of original data')
        plt.axis('equal')
        plt.tight_layout()
        chart_path = assets_path / "original_source_distribution.png"
        plt.savefig(chart_path)
        print(f"\nSource distribution pie chart saved to: {chart_path}")
        
        # 保存source_counts到json文件
        with open(source_counts_file, 'w', encoding='utf-8') as f:
            json.dump(dict(source_counts), f, ensure_ascii=False, indent=2)
        
        print(f"Source counts saved to: {source_counts_file}")

    # ------------------ 2. 进行抽样 ------------------
    # 根据传入的 sample_ratio 计算每个 source 需要抽样的数量
    sample_counts = {}
    for source, count in source_counts.items():
        sample_counts[source] = max(1, int(count * sample_ratio))
    
    # 按比例分层抽样
    sampled_source_counts = Counter()
    current_batch_data = []
    file_count = 0
    total_sampled = 0
    
    print("\nPerforming stratified sampling...")
    for parquet_file in tqdm(parquet_files, desc="Sampling files"):
        # 每个文件的所有分类均按照 sample_ratio 进行抽样
        df = pd.read_parquet(parquet_file)
        
        # 按 source 分组
        for source in source_counts.keys():
            source_data = df[df['source'] == source]
            if len(source_data) == 0:  # 如果该 source 在当前文件中没有数据，则跳过
                continue
            
            # 计算当前文件中这个 source 需要抽样的数量
            source_sample_needed = sample_counts[source]  # 该 source 需要抽样的数量
            source_sampled_so_far = sampled_source_counts[source]  # 该 source 已经抽样的数量
            
            # 按比例计算当前文件中需要抽样的数量
            file_sample_count = max(1, int(len(source_data) * sample_ratio))
            
            # 如果已经抽样够了，就跳过
            if source_sampled_so_far >= source_sample_needed:
                continue
                
            # 调整抽样数量，确保不超过总需求
            file_sample_count = min(file_sample_count, source_sample_needed - source_sampled_so_far)
            
            # 随机抽样
            if file_sample_count < len(source_data):
                sampled_subset = source_data.sample(n=file_sample_count, random_state=123)
            else:
                sampled_subset = source_data
            
            # 将抽样数据添加到当前批次
            current_batch_data.append(sampled_subset)
            sampled_source_counts[source] += len(sampled_subset)
            total_sampled += len(sampled_subset)
            
            # 检查当前批次是否达到最大行数
            current_batch_rows = sum(len(df) for df in current_batch_data)
            if current_batch_rows >= max_rows_per_file:
                # 保存当前批次为文件
                file_count += 1
                file_suffix = f"{file_count:03d}"
                file_path = f"{output_path}/fineweb_edu_sampled_{int(sample_ratio*100)}_percent_{file_suffix}.parquet"
                
                batch_df = pd.concat(current_batch_data, ignore_index=True)
                batch_df.to_parquet(file_path)
                
                file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
                print(f"Saved file {file_count}: {file_path} ({file_size:.2f} GB, {len(batch_df)} rows)")
                
                # 重置当前批次
                current_batch_data = []
    
    # 保存剩余的数据
    if current_batch_data:
        file_count += 1
        file_suffix = f"{file_count:03d}"
        file_path = f"{output_path}/fineweb_edu_sampled_{int(sample_ratio*100)}_percent_{file_suffix}.parquet"
        
        batch_df = pd.concat(current_batch_data, ignore_index=True)
        batch_df.to_parquet(file_path)
        
        file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
        print(f"Saved file {file_count}: {file_path} ({file_size:.2f} GB, {len(batch_df)} rows)")
    
    print(f"\nTotal sampled rows: {total_sampled} (sampling ratio: {total_sampled/sum(source_counts.values())*100:.2f}%)")
    
    # 打印抽样后source分布
    print("\nSource distribution of sampled data:")
    sampled_total = sum(sampled_source_counts.values())
    for source, count in sampled_source_counts.most_common():
        percentage = (count / sampled_total) * 100
        print(f"    {source}: {count} ({percentage:.2f}%)")
    
    # 绘制抽样后source分布图
    plt.figure(figsize=(10, 8))
    sources = list(sampled_source_counts.keys())
    counts = list(sampled_source_counts.values())
    
    # 创建饼状图
    plt.pie(counts, labels=sources, autopct='%1.1f%%', startangle=90)
    plt.title(f'Source distribution of sampled data (sampling ratio: {sample_ratio*100}%)')
    plt.axis('equal')
    plt.tight_layout()
    chart_path = f"{output_path}/sampled_source_distribution.png"
    plt.savefig(chart_path)
    print(f"\nSource distribution pie chart saved to: {chart_path}")
    
    return None


# ------------------------------------- 数据拼接函数 -------------------------------------
def merge_pretrain_data(
    merge_list: list,
    bin_path: str = str(processed_pretrain_data_path),
    pretrain_data_path: str = str(pretrain_data_path)
    ):
    """
    合并多个预处理好的二进制文件为一个文件
    
    Args:
        merge_list (list): 需要合并的文件名列表（不包含路径）
        bin_path (str): 二进制文件所在目录路径
        pretrain_data_path (str): 合并后文件保存的目录路径
    """
    print("Starting to merge pretrain data files...")
    
    # 输出文件路径
    output_file = f"{pretrain_data_path}/pretrain_data.bin"
    
    # 统计总token数
    total_tokens = 0
    
    with open(output_file, 'wb') as out_f:
        for filename in tqdm(merge_list, desc="Merging pretrain files"):
            file_path = Path(bin_path) / filename
            
            if not file_path.exists():
                print(f"Warning: File {file_path} does not exist, skipping...")
                continue
                
            # 读取二进制文件
            with open(file_path, 'rb') as in_f:
                # 读取整个文件内容
                content = in_f.read()
                # 写入到输出文件
                out_f.write(content)
                
                # 计算token数量
                token_count = len(content) // 2
                total_tokens += token_count
    
    print("-" * 30)
    print(f"Pretrain data merging completed!")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Output file: {output_file}")
    print(f"Output file size: {os.path.getsize(output_file) / (1024*1024*1024):.2f} GB")
    print("-" * 30)


if __name__ == "__main__":
    
    # 根据需要处理数据集，无需处理的可以注释掉
    print("=" * 30)
    print("Start processing pretrain datasets...")
    print("=" * 30)

    # --------------------------------- 处理 pretrain 数据集 ---------------------------------
    # step 1. 按比例分层抽样 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集
    # sample_ratio = 0.2
    # fineweb_edu_sampled_output_path = str(pretrain_data_path / f"fineweb_edu_sampled_{int(sample_ratio*100)}_percent")
    # analyze_and_sample_fineweb_edu(
    #     data_path=str(fineweb_edu_file_path), 
    #     output_path=str(fineweb_edu_sampled_output_path), 
    #     sample_ratio=sample_ratio
    #     )
    
    # step 2. 对数据集进行分词处理
    # 分词处理抽样后的 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集
    # process_pretrain_fineweb_edu(
    #     data_path=str(fineweb_edu_sampled_output_path), 
    #     bin_path=str(processed_pretrain_data_path / f"fineweb_edu_sampled_{int(sample_ratio*100)}_percent.bin")
    #     )

    # 分词处理原 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集
    # process_pretrain_fineweb_edu(data_path=str(fineweb_edu_file_path), bin_path=str(fineweb_edu_bin_path))

    # 分词处理 deepctrl 数据集
    # 对于小模型而言，可以先在 sft 任务数据上进行 next token prediction 的训练，然后再微调学习对话格式
    # minimind 项目的预训练、nanochat 的 mid-training 就是这种策略
    # process_pretrain_deepctrl(data_path=str(deepctrl_file_path), bin_path=str(deepctrl_bin_path))

    # ------------------------------------- 合并多个数据集 ------------------------------------
    # step 3. 如果想要训练多个不同的数据集，可以根据需要将他们拼接起来
    # 如果只有一个数据集需要训练,可以无需合并,直接在训练时传入该数据集路径即可
    merge_pretrain_data(merge_list=["deepctrl.bin", "fineweb_edu.bin"])