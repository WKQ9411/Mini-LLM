from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import json
import pandas as pd
from pathlib import Path
import re
import random
import os
import matplotlib.pyplot as plt
import jsonlines
from collections import defaultdict


# 定义路径
root_path = Path(__file__).parent.parent
# sft
sft_data_path = root_path / "data/sft_data"
processed_sft_data_path = sft_data_path / "jsonl"
processed_sft_data_path.mkdir(parents=True, exist_ok=True)
parquet_sft_data_path = sft_data_path / "parquet"
parquet_sft_data_path.mkdir(parents=True, exist_ok=True)

# 加载训练好的分词器路径
tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))
eos_token = tokenizer.eos_token
pad_token = tokenizer.pad_token


# ----------------------------------------- 数据集路径 -----------------------------------------
# deepctrl 数据集 - sft
# https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data
deepctrl_file_path = sft_data_path / "deepctrl/sft_data_zh.jsonl"
deepctrl_jsonl_path = processed_sft_data_path / "deepctrl.jsonl"

# 自我认知数据集 - sft
self_cognition_jsonl_path = processed_sft_data_path / "self_cognition.jsonl"


# ------------------------------------- sft data 构造 -------------------------------------
# 处理 DeepCtrl 数据集
def process_sft_deepctrl(
    data_path: str, 
    jsonl_path: str, 
    num_samples: int | None = None, 
    min_seq_len: int = 20, 
    max_seq_len: int = 512, 
    multi_turn_ratio: float = 0.2
    ):
    """
    读取 data 文件, 筛选出符合长度要求的样本, 构造为 message 格式, 保存为 jsonl 文件

    Args:
        data_path (str): 输入的 data 文件路径
        jsonl_path (str): 输出的 jsonl 文件路径
        num_samples (int | None): 筛选的样本数量, 默认为 None, 代表从原始 11,381,621 条数据中筛选出所有符合条件的数据
        min_seq_len (int): 最小序列长度, 按照 token id 长度计算
        max_seq_len (int): 最大序列长度, 按照 token id 长度计算
        multi_turn_ratio (float): 多轮对话样本比例, 用于筛选多轮对话样本, 根据数据集的不同和最大序列长度限制, 此比例为最大比例
    """
    print(f"Start processing SFT data: {data_path}")
    
    # 存储处理后的数据，按type和是否多轮分类
    single_turn_by_type = defaultdict(list)  # 按type分类的单轮数据
    multi_turn_by_type = defaultdict(list)   # 按type分类的多轮数据
    
    # 先计算总行数
    print("Calculating total lines of the file. This may take a while for large files ...")
    with open(data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        print(f"Total lines: {total_lines:,}")
    
    # 读取JSONL文件
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing SFT data"):
            data = json.loads(line.strip())
            
            # 构造message格式
            messages = []
            
            # 如果有history，先添加历史对话
            skip_item = False
            if data.get('history'):
                for turn in data['history']:
                    # 确保历史对话有实际内容
                    if turn[0] and turn[1]:
                        messages.append({"role": "user", "content": turn[0]})
                        messages.append({"role": "assistant", "content": turn[1]})
                    else:
                        skip_item = True
                        break
                # 如果history中有无效数据，跳过整个数据项
                if skip_item:
                    continue
            
            # 添加当前对话，仅当instruction、input、output都有内容，或仅input、output有内容时才添加
            if data['instruction'] and data['input'] and data['output']:
                messages.append({"role": "user", "content": data['instruction'] + data['input']})
                messages.append({"role": "assistant", "content": data['output']})
            elif data['input'] and data['output']:
                messages.append({"role": "user", "content": data['input']})
                messages.append({"role": "assistant", "content": data['output']})
            else:
                continue
            
            # 使用聊天模板格式化文本
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # 先使用文本长度进行快速过滤，文本长度区间放大 20%
            if len(formatted_text) < min_seq_len * 0.8 or len(formatted_text) > max_seq_len * 1.2:
                continue

            # 文本区间缩小 20%，即只对在边界附近的文本进行 token id 长度精细过滤，如果在边界中很安全的距离，就不用做 tokenizer
            if len(formatted_text) < min_seq_len * 1.2 or len(formatted_text) > max_seq_len * 0.8:
                total_tokens = len(tokenizer.encode(formatted_text))
                if total_tokens < min_seq_len or total_tokens > max_seq_len:
                    continue
            
            # 获取type
            data_type = data.get('type', "unknown")
            
            # 根据是否有history分类存储
            if data.get('history'):
                multi_turn_by_type[data_type].append({"messages": messages, "type": data_type})
            else:
                single_turn_by_type[data_type].append({"messages": messages, "type": data_type})
    
    # 统计有效数据
    total_single_turn = sum(len(samples) for samples in single_turn_by_type.values())
    total_multi_turn = sum(len(samples) for samples in multi_turn_by_type.values())
    total_filtered = total_single_turn + total_multi_turn
    
    if total_filtered == 0:
        print("There is no data that meets the requirements!")
        return
    
    print(f"Filtered data: {total_filtered:,} samples (single-turn: {total_single_turn:,}, multi-turn: {total_multi_turn:,})")
    
    # 判断是否需要抽样
    if num_samples is None:
        # 不进行抽样，使用所有符合条件的数据
        print("Using all filtered data (no sampling)")
        sampled_single_turn = []
        sampled_multi_turn = []
        
        # 收集所有单轮数据
        for type_name, samples in single_turn_by_type.items():
            sampled_single_turn.extend(samples)
            
        # 收集所有多轮数据
        for type_name, samples in multi_turn_by_type.items():
            sampled_multi_turn.extend(samples)
            
        # 打乱顺序
        random.shuffle(sampled_single_turn)
        random.shuffle(sampled_multi_turn)
    else:
        # 计算目标样本数量
        target_single_turn = int(num_samples * (1 - multi_turn_ratio))
        target_multi_turn = num_samples - target_single_turn
        
        print(f"Target samples: {num_samples:,} (single-turn: {target_single_turn:,}, multi-turn: {target_multi_turn:,})")
        
        # 按type均衡抽样
        def balanced_sampling(data_by_type, target_count):
            """按type均衡抽样"""
            if not data_by_type or target_count <= 0:
                return []
            
            # 获取所有type
            types = list(data_by_type.keys())
            num_types = len(types)
            
            # 计算每个type的目标样本数
            samples_per_type = max(1, target_count // num_types)
            remaining_samples = target_count - (samples_per_type * num_types)
            
            result = []
            
            # 为每个type抽样
            for i, type_name in enumerate(types):
                type_samples = data_by_type[type_name]
                
                # 前面的type多抽一个样本，直到分配完剩余样本
                current_samples_per_type = samples_per_type + (1 if i < remaining_samples else 0)
                
                # 如果该type的样本不足，则全部取用
                if len(type_samples) <= current_samples_per_type:
                    result.extend(type_samples)
                else:
                    result.extend(random.sample(type_samples, current_samples_per_type))
            
            # 如果样本还不够，从样本较多的type中补充
            if len(result) < target_count:
                needed = target_count - len(result)
                
                # 收集所有还有剩余样本的type
                remaining_types = []
                for type_name in types:
                    used_count = min(len(data_by_type[type_name]), samples_per_type + (1 if types.index(type_name) < remaining_samples else 0))
                    remaining = data_by_type[type_name][used_count:]
                    if remaining:
                        remaining_types.extend(remaining)
                
                # 随机抽取剩余样本
                if remaining_types:
                    result.extend(random.sample(remaining_types, min(needed, len(remaining_types))))
            
            return result[:target_count]  # 确保不超过目标数量
        
        # 分别对单轮和多轮数据进行均衡抽样
        sampled_single_turn = balanced_sampling(single_turn_by_type, target_single_turn)
        sampled_multi_turn = balanced_sampling(multi_turn_by_type, target_multi_turn)
    
    # 合并数据
    final_data = sampled_single_turn + sampled_multi_turn
    random.shuffle(final_data)  # 打乱顺序
    
    # 写入jsonl文件，只保留messages字段
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps({"messages": item["messages"]}, ensure_ascii=False) + '\n')
    
    print("-" * 30)
    print(f"Processing completed!")
    print(f"Total write {len(final_data):,} samples to: {jsonl_path}")
    print(f"Num of single-turn data: {len(sampled_single_turn):,}")
    print(f"Num of multi-turn data: {len(sampled_multi_turn):,}")
    print(f"Final multi-turn ratio: {len(sampled_multi_turn)/len(final_data):.2%}")
    print("-" * 30)


# 生成自我认知数据集
def process_self_cognition(
    jsonl_path: str, 
    model_name: str = "Mini-LLM", 
    owner: str = "WKQ", 
    batch: int = 10 , 
    num_samples: int = 50, 
    max_seq_len: int = 512, 
    async_num: int = 10
    ):
    """
    调用 OpenAI 兼容的 API, 构造为 message 格式的 sft jsonl 文件, 用于模型自我认知

    Args:
        jsonl_path (str): 输出的 jsonl 文件路径
        model_name (str): 期望的模型名称
        owner (str): 期望的模型所有者
        batch (int): 每次请求生成的样本数量
        num_samples (int): 总共需要生成的样本数量
        max_seq_len (int): 最大序列长度, 按照 token id 长度计算
        async_num (int): 异步请求数量, 用于控制并发请求数量
    """
    import asyncio
    from openai import AsyncOpenAI
    from dotenv import load_dotenv
    import json_repair
    
    # 加载环境变量
    load_dotenv()
    
    # 检查必要的环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_id = os.getenv("OPENAI_MODEL_ID")
    
    if not api_key or not base_url or not model_id:
        print("Error: OPENAI_API_KEY, OPENAI_BASE_URL, or OPENAI_MODEL_ID environment variable is not set")
        print("Please set in .env file: OPENAI_API_KEY=your_api_key, OPENAI_BASE_URL=your_base_url, OPENAI_MODEL_ID=your_model_id")
        return
    
    # 初始化异步客户端
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 自我认知数据生成提示词
    prompt = """你是一个专业的AI助手数据生成器。请为模型"{model_name}"（由{owner}开发）生成{batch}条自我认知相关的对话数据，上下文长度不超过{max_seq_len}个token。
    
要求：
1. 生成多样化的问答对，涵盖模型身份、开发者、能力、特点等不同角度
2. 回答要自然、准确，符合模型的实际情况
3. 每条数据都要以可解析的JSON格式输出，包含"messages"字段
4. "messages"字段是一个数组，包含user和assistant角色的对话
5. 直接输出JSON数组，不要有任何其他解释或说明
6. {model_name} 是一个旨在复现当前主流开源LLM架构，在较小算力上实现的一个迷你版语言模型，{owner} 是该模型的开发者

输出格式示例：
[
  {{
    "messages": [
      {{"role": "user", "content": "你叫什么名字？"}},
      {{"role": "assistant", "content": "我是{model_name}，一个由{owner}开发的语言模型。"}}
    ]
  }},
  ...
]"""
    prompt = prompt.format(model_name=model_name, owner=owner, batch=batch, max_seq_len=max_seq_len)
    
    async def generate_data():
        all_results = []
        
        # 定义单个异步请求函数
        async def fetch_batch():
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1.2,
                    max_tokens=6000
                )
                
                content = response.choices[0].message.content
                # 尝试解析JSON
                try:
                    # 提取JSON数组部分
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        batch_data = json_repair.loads(json_str)
                    else:
                        # 如果没有找到JSON数组，尝试直接解析
                        batch_data = json_repair.loads(content)
                    
                    # 验证并过滤数据
                    valid_data = []
                    for item in batch_data:
                        if "messages" in item and isinstance(item["messages"], list):
                            # 使用tokenizer检查长度
                            formatted_text = tokenizer.apply_chat_template(
                                item["messages"], 
                                tokenize=False,
                                add_generation_prompt=False
                            )
                            tokenized = tokenizer(formatted_text)
                            
                            if len(tokenized["input_ids"]) <= max_seq_len:
                                valid_data.append(item)
                    
                    return valid_data
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing response: {e}")
                    print(f"Response content: {content[:200]}...")
                    return []
                    
            except Exception as e:
                print(f"Error in API request: {e}")
                return []
        
        # 循环直到获得足够的样本
        with tqdm(total=num_samples, desc="Generating self-cognition data") as pbar:
            # 创建异步任务队列
            tasks = []
            
            # 初始创建一批任务
            for _ in range(min(async_num, (num_samples + batch - 1) // batch)):
                tasks.append(fetch_batch())
            
            # 执行任务并循环
            while len(all_results) < num_samples:
                # 逐个执行任务，以便实时更新进度条
                for i, task in enumerate(asyncio.as_completed(tasks)):
                    results = await task
                    if results:
                        all_results.extend(results)
                        pbar.update(len(results))
                        # 实时刷新进度条显示
                        pbar.refresh()
                
                # 如果已经获得足够的样本，退出循环
                if len(all_results) >= num_samples:
                    break
                
                # 计算还需要多少样本
                remaining = num_samples - len(all_results)
                # 计算需要多少批次
                batches_needed = (remaining + batch - 1) // batch
                # 创建新的任务
                tasks = [fetch_batch() for _ in range(min(async_num, batches_needed))]
        
        # 截取到所需数量
        all_results = all_results[:num_samples]
        
        # 写入JSONL文件
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print("-" * 30)
        print(f"Self-cognition data generation completed!")
        print(f"Successfully generated {len(all_results):,} samples")
        print(f"Data saved to: {jsonl_path}")
        print("-" * 30)
    
    # 运行异步生成函数
    asyncio.run(generate_data())


# ------------------------------------- 数据拼接函数 -------------------------------------
def merge_and_convert_to_parquet(
    merge_list: list,
    jsonl_path: str = str(processed_sft_data_path), 
    parquet_path: str = str(parquet_sft_data_path),
    max_seq_len: int = 512,
    ignore_index: int = -100
    ):
    """
    合并多个 sft 数据集的 jsonl 文件并转换保存为 parquet 文件
    该 parquet 文件包括 token_ids, labels, position_ids, length, attention_mask, type 字段
    
    Args:
        merge_list (list): 需要合并的文件名列表
        jsonl_path (str): jsonl 文件所在目录路径
        parquet_path (str): 合并并转换后 parquet 文件保存的目录路径
        max_seq_len (int): 最大序列长度, 默认为 512
        ignore_index (int): 用于 padding 的 ignore index, 默认为 -100
    """
    print("Starting to merge SFT jsonl files and convert to parquet...")
    
    # 确保输出目录存在
    Path(parquet_path).mkdir(parents=True, exist_ok=True)
    output_file = f"{parquet_path}/sft_data.parquet"
    
    # 收集所有数据，同时记录来源文件名
    all_data = []
    
    # 读取所有 JSONL 文件
    for filename in tqdm(merge_list, desc="Reading JSONL files"):
        file_path = Path(jsonl_path) / filename
        
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        # 读取 JSONL 文件
        with open(file_path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                # 验证JSON格式
                try:
                    data = json.loads(line.strip())
                    # 确保包含messages字段
                    if "messages" in data:
                        # 添加来源文件名信息
                        data['_source_file'] = filename
                        all_data.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON line in {filename}, skipping...")
                    continue
    
    print(f"Total samples read: {len(all_data):,}")
    print("Processing samples and converting to parquet format...")
    
    # 处理每条数据，生成所需字段
    processed_data = []
    
    for item in tqdm(all_data, desc="Processing samples"):
        messages = item['messages']
        source_file = item.get('_source_file', '')
        
        # 确定 type 字段
        if source_file == 'self_cognition.jsonl':
            data_type = 'self_cognition'
        else:
            # 对于其他文件，根据 message 内字典数量判断
            message_count = len(messages)
            if message_count == 2:
                data_type = 'single_turn'
            elif message_count > 2:
                data_type = 'multi_turn'
            else:
                # 如果 message 数量小于 2，默认为 single_turn
                data_type = 'single_turn'
        
        # 1. 完整对话 token
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # 2. 计算 prompt 部分的长度进行 mask
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        
        # 3. 构建 labels, prompt 部分设为 ignore_index
        labels = [ignore_index] * prompt_len + full_ids[prompt_len:]
        assert len(full_ids) == len(labels), "full_ids and labels must have the same length"
        
        # 4. 截断
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            labels = labels[:max_seq_len]
        
        # 5. padding
        seq_len = len(full_ids)
        pad_len = max_seq_len - seq_len
        
        attention_mask = [1] * seq_len + [0] * pad_len
        token_ids = full_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [ignore_index] * pad_len
        
        # 6. position ids, 简单递增
        position_ids = list(range(max_seq_len))
        
        # 7. length， 记录 padding 前的 token id 长度
        length = seq_len
        
        processed_data.append({
            "token_ids": token_ids,
            "labels": labels,
            "position_ids": position_ids,
            "length": length,
            "attention_mask": attention_mask,
            "type": data_type
        })
    
    # 转换为 DataFrame 并保存为 parquet
    df = pd.DataFrame(processed_data)
    df.to_parquet(output_file, index=False)
    
    print("-" * 30)
    print(f"SFT data merging and conversion completed!")
    print(f"Total samples: {len(processed_data):,}")
    print(f"Output file: {output_file}")
    print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print("-" * 30)


def analyze_sft_data(parquet_data_path: str = str(parquet_sft_data_path / "sft_data.parquet"), interval: int = 50):
    """
    分析 parquet 文件数据的长短分布情况, 并保存为条形图
    
    Args:
        parquet_data_path (str): parquet 文件路径
        interval (int): 统计区间大小, 默认为 50
    """
    print(f"Analyzing SFT data length distribution: {parquet_data_path}")
    
    # 从 parquet 文件读取数据
    print("Reading parquet file...")
    df = pd.read_parquet(parquet_data_path)
    
    # 直接从 length 字段获取长度数据
    if 'length' not in df.columns:
        raise ValueError("The parquet file does not contain a 'length' column")
    
    lengths = df['length'].tolist()
    
    # 计算统计信息
    min_length = int(df['length'].min())
    max_length = int(df['length'].max())
    avg_length = float(df['length'].mean())
    
    print(f"Sequence Length Statistics:")
    print(f"  Minimum Length: {min_length}")
    print(f"  Maximum Length: {max_length}")
    print(f"  Average Length: {avg_length:.2f}")
    
    # 按区间统计
    length_distribution = defaultdict(int)
    for length in lengths:
        # 计算所属区间
        bin_index = length // interval
        bin_start = bin_index * interval
        bin_end = bin_start + interval - 1
        bin_label = f"{bin_start}-{bin_end}"
        length_distribution[bin_label] += 1
    
    # 按区间排序
    sorted_bins = sorted(length_distribution.items(), key=lambda x: int(x[0].split('-')[0]))
    bin_labels = [item[0] for item in sorted_bins]
    bin_counts = [item[1] for item in sorted_bins]
    
    # 绘制条形图
    plt.figure(figsize=(15, 8))
    bars = plt.bar(bin_labels, bin_counts, color='skyblue', edgecolor='black')
    
    # 添加数值标签
    for bar, count in zip(bars, bin_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bin_counts) * 0.01,
                 str(count), ha='center', va='bottom')
    
    # 设置图表标题和标签
    plt.title(f'Sequence Length Distribution (Interval: {interval})', fontsize=16)
    plt.xlabel('Sequence Length Interval', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(parquet_data_path).parent / f"{Path(parquet_data_path).stem}_length_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Sequence length distribution chart saved to: {output_path}")
    
    # 关闭图表，释放内存
    plt.close()
    
    # 返回统计结果
    return {
        "total_samples": len(lengths),
        "min_length": min_length,
        "max_length": max_length,
        "avg_length": avg_length,
        "length_distribution": dict(sorted_bins)
    }


def sample_from_sft_parquet(
    sft_parquet_path: str,
    sampled_sft_parquet_path: str,
    sample_num: dict
    ):
    """
    从 sft_data.parquet 文件中按 type 字段进行抽样，生成 sampled_sft_data.parquet 文件
    
    Args:
        sft_parquet_path (str): 输入的 parquet 文件路径
        sampled_sft_parquet_path (str): 输出的抽样后 parquet 文件路径
        sample_num (dict): 每个 type 需要抽样的数量，例如 {"single_turn": 40000, "multi_turn": 10000, "self_cognition": 200}
    """
    print("Starting to sample from SFT parquet file...")
    print(f"Input file: {sft_parquet_path}")
    print(f"Output file: {sampled_sft_parquet_path}")
    print(f"Sample numbers: {sample_num}")
    
    # 确保输出目录存在
    output_path = Path(sampled_sft_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取 parquet 文件
    print("Reading parquet file...")
    df = pd.read_parquet(sft_parquet_path)
    
    # 检查是否有 type 字段
    if 'type' not in df.columns:
        raise ValueError("The parquet file does not contain a 'type' column")
    
    print(f"Total samples in input file: {len(df):,}")
    
    # 统计每个 type 的数量
    type_counts = df['type'].value_counts().to_dict()
    print("\nType distribution in input file:")
    for type_name, count in sorted(type_counts.items()):
        print(f"  {type_name}: {count:,}")
    
    # 按 type 进行抽样
    sampled_dfs = []
    
    for type_name, target_num in sample_num.items():
        if type_name not in type_counts:
            print(f"Warning: Type '{type_name}' not found in the data, skipping...")
            continue
        
        available_count = type_counts[type_name]
        
        # 获取该 type 的所有数据
        type_df = df[df['type'] == type_name].copy()
        
        # 如果可用数量小于目标数量，使用全部数据
        if available_count <= target_num:
            print(f"  {type_name}: Using all {available_count:,} samples (requested: {target_num:,})")
            sampled_dfs.append(type_df)
        else:
            # 随机抽样
            sampled_type_df = type_df.sample(n=target_num, random_state=None)
            print(f"  {type_name}: Sampled {target_num:,} from {available_count:,} samples")
            sampled_dfs.append(sampled_type_df)
    
    # 合并所有抽样结果
    if not sampled_dfs:
        print("Error: No data sampled!")
        return
    
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # 打乱顺序
    sampled_df = sampled_df.sample(frac=1, random_state=None).reset_index(drop=True)
    
    # 保存到 parquet 文件
    sampled_df.to_parquet(sampled_sft_parquet_path, index=False)
    
    # 统计抽样后的分布
    sampled_type_counts = sampled_df['type'].value_counts().to_dict()
    
    print("\n" + "-" * 30)
    print("Sampling completed!")
    print(f"Total sampled samples: {len(sampled_df):,}")
    print(f"Output file: {sampled_sft_parquet_path}")
    print(f"Output file size: {os.path.getsize(sampled_sft_parquet_path) / (1024*1024):.2f} MB")
    print("\nSampled type distribution:")
    for type_name, count in sorted(sampled_type_counts.items()):
        print(f"  {type_name}: {count:,}")
    print("-" * 30)


def pack_sft_parquet(
    sft_parquet_path: str,
    packed_sft_parquet_path: str,
    max_seq_len: int = 512,
    ignore_index: int = -100
    ):
    """
    对 sft_data.parquet 文件进行 packing，将多个短序列打包到一个序列中，提高训练效率
    
    采用 greedy packing 策略：
    1. 按长度从大到小排序
    2. 依次将样本放入 bin 中，尽量填满每个 bin
    3. 构建拼接后的 token_ids, labels, position_ids
    
    Args:
        sft_parquet_path (str): 输入的 parquet 文件路径
        packed_sft_parquet_path (str): 输出的 packing 后 parquet 文件路径
        max_seq_len (int): 最大序列长度, 默认为 512
        ignore_index (int): 用于 padding 的 ignore index, 默认为 -100
    """
    print("Starting to pack SFT parquet file...")
    print(f"Input file: {sft_parquet_path}")
    print(f"Output file: {packed_sft_parquet_path}")
    print(f"Max sequence length: {max_seq_len}")
    
    # 确保输出目录存在
    output_path = Path(packed_sft_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取 parquet 文件
    print("Reading parquet file...")
    df = pd.read_parquet(sft_parquet_path)
    
    # 检查必要的字段
    required_fields = ['token_ids', 'labels', 'length', 'type']
    for field in required_fields:
        if field not in df.columns:
            raise ValueError(f"The parquet file does not contain a '{field}' column")
    
    print(f"Total samples in input file: {len(df):,}")
    
    # 1. 预处理所有数据，提取有效长度（去除 padding）
    all_samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing samples"):
        token_ids = row['token_ids']
        labels = row['labels']
        length = row['length']  # 使用 length 字段，这是 padding 前的实际长度
        data_type = row['type']
        
        # 只取有效部分（去除 padding）
        valid_token_ids = token_ids[:length]
        valid_labels = labels[:length]
        
        # 如果单条数据超过 max_seq_len，则截断
        if length > max_seq_len:
            valid_token_ids = valid_token_ids[:max_seq_len]
            valid_labels = valid_labels[:max_seq_len]
            length = max_seq_len
        
        all_samples.append({
            "token_ids": valid_token_ids,
            "labels": valid_labels,
            "length": length,
            "type": data_type
        })
    
    # 2. 按长度从大到小排序
    all_samples.sort(key=lambda x: x['length'], reverse=True)
    
    # 3. 开始 greedy packing
    print("Packing samples...")
    bins = []  # 每个 bin 是一个字典，存储该 bin 当前的 list of samples 和 current_len
    
    for sample in tqdm(all_samples, desc="Packing"):
        placed = False  # 当前样本是否已成功放入 bin
        
        # 尝试放入现有的 bin 中
        for bin in bins:
            if bin['current_len'] + sample['length'] <= max_seq_len:
                bin['samples'].append(sample)
                bin['current_len'] += sample['length']
                placed = True
                break
        
        # 如果所有 bin 都放不下，或者还没有 bin，创建一个新的
        if not placed:
            bins.append({
                'samples': [sample],
                'current_len': sample['length']
            })
    
    print(f"Packed {len(all_samples):,} samples into {len(bins):,} bins")
    print(f"Average samples per bin: {len(all_samples) / len(bins):.2f}")
    
    # 4. 将 bin 转换为最终格式
    packed_data = []
    total_pad_token = 0
    
    for bin in tqdm(bins, desc="Processing bins"):
        # 拼接该 bin 内的所有样本
        bin_token_ids = []
        bin_labels = []
        position_ids = []
        bin_types = []  # 记录每个样本的 type
        
        for sample in bin['samples']:
            bin_token_ids.extend(sample['token_ids'])
            bin_labels.extend(sample['labels'])
            bin_types.append(sample['type'])
            # 生成每一段独立的 position ids (0, 1, 2...)
            position_ids.extend(list(range(sample['length'])))
        
        # 如果不满 max_seq_len，则进行 padding
        seq_len = len(bin_token_ids)
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            total_pad_token += pad_len
            bin_token_ids += [tokenizer.pad_token_id] * pad_len
            bin_labels += [ignore_index] * pad_len
            # position ids 对于 padding 部分可以随意，这里简单全0
            position_ids += [0] * pad_len
        
        # attention_mask: 简单的 1D mask，1 表示有效位置，0 表示 padding
        # 注意：2D 的斜对角块 mask 将在 dataset 中构建
        attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        
        # 保存每个样本的长度，用于后续在 dataset 中构建 2D attention mask
        sample_lengths = [sample['length'] for sample in bin['samples']]
        
        # 对于 type，如果 bin 中有多个样本，可以保存主要类型或所有类型
        # 这里保存主要类型（第一个样本的类型，或者可以统计最常见的类型）
        main_type = bin_types[0] if bin_types else "unknown"
        
        packed_data.append({
            "token_ids": bin_token_ids,
            "labels": bin_labels,
            "position_ids": position_ids,
            "length": seq_len,
            "attention_mask": attention_mask,
            "type": main_type,
            "num_samples": len(bin['samples']),  # 记录该 bin 包含的样本数量，用于统计
            "sample_lengths": sample_lengths  # 记录每个样本的长度，用于构建 2D attention mask
        })
    
    # 5. 转换为 DataFrame 并保存
    packed_df = pd.DataFrame(packed_data)
    packed_df.to_parquet(packed_sft_parquet_path, index=False)
    
    # 统计信息
    pad_ratio = total_pad_token / (len(packed_data) * max_seq_len)
    avg_samples_per_bin = packed_df['num_samples'].mean()
    
    print("\n" + "-" * 30)
    print("Packing completed!")
    print(f"Total bins: {len(packed_data):,}")
    print(f"Average samples per bin: {avg_samples_per_bin:.2f}")
    print(f"Total padding tokens: {total_pad_token:,}")
    print(f"Padding ratio: {pad_ratio:.2%}")
    print(f"Output file: {packed_sft_parquet_path}")
    print(f"Output file size: {os.path.getsize(packed_sft_parquet_path) / (1024*1024):.2f} MB")
    print("-" * 30)


if __name__ == "__main__":
    
    # 根据需要处理数据集
    print("=" * 30)
    print("Start processing sft datasets...")
    print("=" * 30)
    
    # --------------------------------- 处理 sft 数据集 ---------------------------------
    # step 1. 处理 deepctrl 数据集
    # 默认筛选出所有符合条件的样本，保存为 jsonl 文件
    # process_sft_deepctrl(data_path=str(deepctrl_file_path), jsonl_path=str(deepctrl_jsonl_path))

    # step 2. 生成自我认知数据集
    # process_self_cognition(jsonl_path=str(self_cognition_jsonl_path), model_name="Mini-LLM", owner="WKQ", num_samples=200)

    # ------------------------------------- 合并多个数据集 ------------------------------------
    # step 3. 合并文件
    # 默认合并形成全部符合条件的样本，保存为 sft_data.parquet 文件，其中的样本已经过 tokenize 处理
    # 如果需调整 sft 训练量，仅需从 sft_data.parquet 文件中随机抽取样本即可
    merge_and_convert_to_parquet(merge_list=["deepctrl.jsonl", "self_cognition.jsonl"])

    # ------------------------------------- 抽样函数 ------------------------------------
    # 可以选择从 sft_data.parquet 中按 type 进行抽样，形成 sampled_sft_data.parquet 文件
    sample_from_sft_parquet(
        sft_parquet_path=str(parquet_sft_data_path / "sft_data.parquet"), 
        sampled_sft_parquet_path=str(parquet_sft_data_path / "sampled_sft_data.parquet"),
        sample_num = {"single_turn": 40000, "multi_turn": 10000, "self_cognition": 200}
        )

    # ------------------------------------- packing 函数 ------------------------------------
    # 形成经过 packing 的 parquet 文件，采用分片 packing 策略提高效率
    pack_sft_parquet(
        sft_parquet_path=str(parquet_sft_data_path / "sampled_sft_data.parquet"),
        packed_sft_parquet_path=str(parquet_sft_data_path / "packed_sft_data.parquet")
        )
    
    # ------------------------------------- 分析 sft 数据长度分布 ------------------------------------
    # 分析合并的、抽样的 sft 数据长度分布
    analyze_sft_data(parquet_data_path=str(parquet_sft_data_path / "sft_data.parquet"), interval=50)
    analyze_sft_data(parquet_data_path=str(parquet_sft_data_path / "sampled_sft_data.parquet"), interval=50)