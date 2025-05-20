from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import json
import os
import pandas as pd
import jsonlines
import re
import csv


# 定义预训练数据路径
# 序列猴子数据集 - pretrain
seq_monkey_file_path = './preprocess_data/data/pretrain_data/mobvoi_seq_monkey_general_open_corpus.jsonl'  # https://github.com/mobvoi/seq-monkey-data/blob/main/docs/pretrain_open_corpus.md
# 维基百科数据集 - pretrain / tokenizer
wikipedia_cn_file_path = './preprocess_data/data/pretrain_data/wikipedia-cn-20230720-filtered.json'  # https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/tree/main
# 匠数科技数据集 - sft
deepctrl_file_path = './preprocess_data/data/sft_data/sft_data_zh.jsonl'  # https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/files

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


# 计算样本中午占比
def chinese_ratio(text: str) -> float:
    """计算文本中中文字符的比例。"""
    if not text:
        return 0
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) / len(text)

# 处理匠数sft数据集
def process_deepctrl(jsonl_path: str, csv_path: str, contain_history: bool = False, chunk_size: int = 1000):
    """
    处理 SFT 数据，提取输入、输出和可选的历史对话，并写入 CSV

    Args:
        jsonl_path (str): 输入 JSON Lines 文件路径。
        csv_path (str): 输出 CSV 文件路径。
        contain_history (bool, optional): 是否包含历史对话。默认为 False。
        chunk_size (int, optional): 每次写入 CSV 的数据块大小。默认为 1000。
    """
    # 如果包含历史，修改输出文件名
    if contain_history:
        dir_name, file_name = os.path.split(csv_path)
        name, ext = os.path.splitext(file_name)
        new_file_name = name + "_history" + ext
        csv_path = os.path.join(dir_name, new_file_name)

    # 准备写入 CSV
    header = ['history', 'q', 'a']
    try:
        # 尝试获取文件总行数以提供准确的进度条
        total_lines = sum(1 for _ in open(jsonl_path, 'rb')) # 使用 'rb' 更快地计数
    except Exception:
        total_lines = None # 如果文件太大或无法读取，则不显示总数

    # 初始化 CSV 文件并写入表头
    pd.DataFrame(columns=header).to_csv(
        csv_path, index=False, lineterminator='\n', quoting=csv.QUOTE_MINIMAL
    )

    processed_count = 0
    valid_chunk_data = []

    try:
        with jsonlines.open(jsonl_path) as reader, \
             tqdm(total=total_lines, desc="Processing lines", unit=" lines") as pbar:

            for idx, obj in enumerate(reader):
                pbar.update(1) # 每次读取一行就更新进度条
                try:
                    # 1. 提取和合并数据 (确保 q/a 至少有其一或 input/output)
                    q = obj.get('input', '') + obj.get('q', '')
                    a = obj.get('output', '') + obj.get('a', '')
                    history_raw = obj.get('history', []) # 默认空列表更安全

                    # 确保 history 是列表，以防万一格式不规范 (如为 null 或字符串)
                    if not isinstance(history_raw, list):
                        history_raw = []

                    # 2. 基本过滤：检查必须字段
                    if not q or not a:
                        continue
                    if contain_history and not history_raw: # 如果需要历史但历史为空，跳过
                        continue

                    # 3. 长度过滤
                    if len(q) < 10 or len(a) < 5:
                        continue

                    history_len = 0
                    if contain_history and history_raw:
                        # 确保 history 内部结构是 [q, a] 对
                        if all(isinstance(item, list) and len(item) == 2 for item in history_raw):
                            history_len = sum(len(h_q) + len(h_a) for h_q, h_a in history_raw)
                        else:
                            # 如果 history 内部结构不规范，可以选择跳过或记录日志
                            # print(f"Skipping line {idx+1}: Invalid history structure.")
                            continue # 跳过此行

                    if history_len + len(q) + len(a) > 450: # 总字符长度过滤
                        continue

                    # 4. 语言过滤 (高中文比例)
                    if not (chinese_ratio(q) > 0.9 and chinese_ratio(a) > 0.9):
                        continue

                    # 5. 构建有效记录
                    valid_record = {
                        'history': history_raw if contain_history else [],
                        'q': q,
                        'a': a
                    }
                    valid_chunk_data.append(valid_record)
                    processed_count += 1

                    # 6. 分块写入 CSV
                    if len(valid_chunk_data) >= chunk_size:
                        df_chunk = pd.DataFrame(valid_chunk_data)
                        df_chunk.to_csv(
                            csv_path, mode='a', header=False, index=False,
                            lineterminator='\n', quoting=csv.QUOTE_MINIMAL
                        )
                        valid_chunk_data = [] # 清空块

                except jsonlines.InvalidLineError:
                    print(f"\nSkipping invalid JSON line {idx + 1}")
                    continue
                except Exception as e: # 捕捉其他潜在错误
                    print(f"\nError processing line {idx + 1}: {e}")
                    continue

            # 处理并写入最后一个不足 chunk_size 的块
            if valid_chunk_data:
                df_chunk = pd.DataFrame(valid_chunk_data)
                df_chunk.to_csv(
                    csv_path, mode='a', header=False, index=False,
                    lineterminator='\n', quoting=csv.QUOTE_MINIMAL
                )

    except FileNotFoundError:
        print(f"错误：输入文件 {jsonl_path} 未找到。")
        return
    except Exception as e:
        print(f"处理过程中发生未预料的错误: {e}")
        return

    print(f"\n数据处理完成！共处理并写入 {processed_count} 条有效数据到 {csv_path}")


if __name__ == "__main__":
    # 根据需要处理数据集
    process_seq_monkey(seq_monkey_file_path, './preprocess_data/data/pretrain_data/pretrain_data.bin')
    # process_wikipedia(wikipedia_cn_file_path, './preprocess_data/data/pretrain_data/pretrain_data.bin')
    # process_deepctrl(deepctrl_file_path, './preprocess_data/data/sft_data/sft_data.csv')