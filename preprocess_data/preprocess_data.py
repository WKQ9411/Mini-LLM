from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import json


# 定义预训练数据路径
seq_monkey_file_path = './preprocess_data/data/pretrain_data/mobvoi_seq_monkey_general_open_corpus.jsonl'  # https://github.com/mobvoi/seq-monkey-data/blob/main/docs/pretrain_open_corpus.md
wikipedia_cn_file_path = './preprocess_data/data/pretrain_data/wikipedia-cn-20230720-filtered.json'  # https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/tree/main

# 加载训练好的分词器路径
tokenizer = AutoTokenizer.from_pretrained("../mini_tokenizer")
bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token

# 处理 seqq monkey 数据集
def process_seq_monkey(jsonl_path: str, bin_path: str, buffer_size: int = 1000000, dtype_str: str = 'uint16'):
    """
    读取 jsonl 文件, 提取文本字段, 分词, 并将 token id 保存为二进制文件

    Args:
        jsonl_path (str): 输入的 jsonl 文件路径。
        bin_path (str): 输出的二进制文件路径。
        buffer_size (int): 写入磁盘前在内存中缓冲的 token id 数量。
        dtype_str (str): 保存 token id 的 numpy 数据类型 ('uint16', 'uint32'等), 'uint16' 适用于词汇量 < 65536 的分词器, 如果词汇量更大，请使用 'uint32'。
    """
    # 选择合适的 NumPy 数据类型
    try:
        dtype = np.dtype(dtype_str)
    except TypeError:
        print(f"错误：无效的 dtype_str '{dtype_str}'。请使用 'uint16', 'uint32' 等。")
        return

    print(f"词汇量大小: {len(tokenizer)}")
    if dtype == np.uint16 and len(tokenizer) > 65535:
        print(f"警告：分词器词汇量大小 ({len(tokenizer)}) 可能超过了 'uint16' 的最大值 (65535)。考虑使用 'uint32'。")
    
    token_buffer = []
    total_tokens = 0

    print(f"开始处理文件: {jsonl_path}")
    try:
        # 获取文件总行数用于进度条
        print("获取文件总行数...")
        with open(jsonl_path, 'r', encoding='utf-8') as f_in:
            total_lines = sum(1 for _ in f_in)
            print(f"文件总行数: {total_lines}")

        with open(jsonl_path, 'r', encoding='utf-8') as f_in, open(bin_path, 'wb') as f_out:

            # 使用 tqdm 显示进度
            for line in tqdm(f_in, total=total_lines, desc="Processing lines"):
                try:
                    # 解析 jsonl 行
                    data = json.loads(line.strip())
                    
                    # 提取文本
                    text = data.get('text')
                    # 分词
                    token_ids = tokenizer.encode(bos_token+text+eos_token)
                    # 添加到缓冲区
                    token_buffer.extend(token_ids)
                    
                    # 如果缓冲区达到大小，则写入文件
                    if len(token_buffer) >= buffer_size:
                        array_to_write = np.array(token_buffer[:buffer_size], dtype=dtype)
                        array_to_write.tofile(f_out)
                        total_tokens += len(array_to_write)
                        token_buffer = token_buffer[buffer_size:] # 保留剩余部分

                except json.JSONDecodeError:
                    print(f"警告：无法解析 jsonl 行: {line.strip()}")
                except Exception as e:
                    print(f"处理行时发生意外错误: {e} - 行内容: {line.strip()}")

            # 处理结束后，写入缓冲区中剩余的 token
            if token_buffer:
                array_to_write = np.array(token_buffer, dtype=dtype)
                array_to_write.tofile(f_out)
                total_tokens += len(array_to_write)
                print(f"写入最后 {len(token_buffer)} 个 tokens。")

    except FileNotFoundError:
        print(f"错误：输入文件未找到 {jsonl_path}")
        return
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return

    print("-" * 30)
    print(f"处理完成!")
    print(f"总共写入 {total_tokens} 个 Token ID 到 {bin_path}")
    print(f"使用的数据类型: {dtype.name}")
    print("-" * 30)


# 处理 wikipedia 数据集
def process_wikipedia(json_path: str, bin_path: str, buffer_size: int = 1000000, dtype_str: str = 'uint16'):
    """
    读取 json 文件, 提取文本字段, 分词, 并将 token id 保存为二进制文件

    Args:
        json_path (str): 输入的 jsonl 文件路径。
        bin_path (str): 输出的二进制文件路径。
        buffer_size (int): 写入磁盘前在内存中缓冲的 token id 数量。
        dtype_str (str): 保存 token id 的 numpy 数据类型 ('uint16', 'uint32'等), 'uint16' 适用于词汇量 < 65536 的分词器, 如果词汇量更大，请使用 'uint32'。
    """
    # 选择合适的 NumPy 数据类型
    try:
        dtype = np.dtype(dtype_str)
    except TypeError:
        print(f"错误：无效的 dtype_str '{dtype_str}'。请使用 'uint16', 'uint32' 等。")
        return

    print(f"词汇量大小: {len(tokenizer)}")
    if dtype == np.uint16 and len(tokenizer) > 65535:
        print(f"警告：分词器词汇量大小 ({len(tokenizer)}) 可能超过了 'uint16' 的最大值 (65535)。考虑使用 'uint32'。")
    
    token_buffer = []
    total_tokens = 0

    print(f"开始处理文件: {json_path}")
    # 获取文件总行数用于进度条
    print("获取文件总行数...")
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        total_lines = len(data)
        print(f"文件总行数: {total_lines}")

    with open(json_path, 'r', encoding='utf-8') as f_in, open(bin_path, 'wb') as f_out:
        total_data = json.load(f_in)
        # 使用 tqdm 显示进度
        for data in tqdm(total_data, total=total_lines, desc="Processing lines"):
            # 提取文本
            text = data.get('completion')
            # 分词
            token_ids = tokenizer.encode(bos_token+text+eos_token)
            # 添加到缓冲区
            token_buffer.extend(token_ids)
            
            # 如果缓冲区达到大小，则写入文件
            if len(token_buffer) >= buffer_size:
                array_to_write = np.array(token_buffer[:buffer_size], dtype=dtype)
                array_to_write.tofile(f_out)
                total_tokens += len(array_to_write)
                token_buffer = token_buffer[buffer_size:] # 保留剩余部分

        # 处理结束后，写入缓冲区中剩余的 token
        if token_buffer:
            array_to_write = np.array(token_buffer, dtype=dtype)
            array_to_write.tofile(f_out)
            total_tokens += len(array_to_write)
            print(f"写入最后 {len(token_buffer)} 个 tokens。")

    print("-" * 30)
    print(f"处理完成!")
    print(f"总共写入 {total_tokens} 个 Token ID 到 {bin_path}")
    print(f"使用的数据类型: {dtype.name}")
    print("-" * 30)


if __name__ == "__main__":
    # 根据需要处理数据集
    process_seq_monkey(seq_monkey_file_path, './preprocess_data/data/pretrain_data/seq_monkey_data.bin')
    process_wikipedia(wikipedia_cn_file_path, './preprocess_data/data/pretrain_data/wiki_data.bin')