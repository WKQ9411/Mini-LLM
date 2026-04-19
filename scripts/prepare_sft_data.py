from tqdm import tqdm
from transformers import AutoTokenizer
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import re
import random
import os
import matplotlib.pyplot as plt
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
    ignore_index: int = -100,
    batch_size: int = 50000
    ):
    """
    合并多个 sft 数据集的 jsonl 文件并转换保存为 parquet 文件
    该 parquet 文件包括 token_ids, labels, position_ids, length, attention_mask, type 字段
    
    使用分批写入策略，减少内存占用: 每处理 batch_size 条数据后写入磁盘并释放内存
    
    Args:
        merge_list (list): 需要合并的文件名列表
        jsonl_path (str): jsonl 文件所在目录路径
        parquet_path (str): 合并并转换后 parquet 文件保存的目录路径
        max_seq_len (int): 最大序列长度, 默认为 512
        ignore_index (int): 用于 padding 的 ignore index, 默认为 -100
        batch_size (int): 每批处理的样本数量, 默认为 50000, 处理完一批后写入磁盘并释放内存
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
    
    # 定义 parquet schema
    schema = pa.schema([
        ('token_ids', pa.list_(pa.int32())),
        ('labels', pa.list_(pa.int32())),
        ('position_ids', pa.list_(pa.int32())),
        ('length', pa.int32()),
        ('attention_mask', pa.list_(pa.int32())),
        ('type', pa.string())
    ])
    
    # 使用 ParquetWriter 进行增量写入
    writer = None
    processed_data = []
    total_samples = 0
    batch_count = 0
    
    def write_batch(data_batch, writer, schema):
        """将一批数据写入 parquet 文件"""
        if not data_batch:
            return writer
        
        # 转换为 PyArrow Table
        table = pa.Table.from_pydict({
            'token_ids': [d['token_ids'] for d in data_batch],
            'labels': [d['labels'] for d in data_batch],
            'position_ids': [d['position_ids'] for d in data_batch],
            'length': [d['length'] for d in data_batch],
            'attention_mask': [d['attention_mask'] for d in data_batch],
            'type': [d['type'] for d in data_batch]
        }, schema=schema)
        
        # 如果是第一批，创建 writer
        if writer is None:
            writer = pq.ParquetWriter(output_file, schema)
        
        # 写入数据
        writer.write_table(table)
        return writer
    
    for item in tqdm(all_data, desc="Processing samples"):
        messages = item['messages']
        source_file = item.get('_source_file', '')
        
        # 确定 type 字段
        if source_file == 'self_cognition.jsonl':
            data_type = 'self_cognition'
        else:
            # 对于其他文件，根据 message 内字典数量判断
            message_count = len(messages)
            # 确保 message 数量为偶数，且 user/assistant 交替出现
            if message_count % 2 != 0:
                continue
            # 检查是否严格交替为 user、assistant
            valid_roles = True
            for idx, msg in enumerate(messages):
                expected_role = "user" if idx % 2 == 0 else "assistant"
                if msg.get("role") != expected_role:
                    valid_roles = False
                    break
            if not valid_roles:
                continue
            if message_count == 2:
                data_type = 'single_turn'
            elif message_count > 2:
                data_type = 'multi_turn'
            else:
                continue
        
        # 完整对话 token_id
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # 1. 两两 message 为一对，对所有 assistant 的回复构造有效 label
        labels = []
        for turn in range(0, len(messages), 2):
            user_msg = messages[turn]
            assistant_msg = messages[turn + 1]

            full_turn_text = tokenizer.apply_chat_template([user_msg, assistant_msg], tokenize=False, add_generation_prompt=False)
            full_turn_ids = tokenizer(full_turn_text, add_special_tokens=False)["input_ids"]

            prompt_text = tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            labels.extend([ignore_index] * prompt_len + full_turn_ids[prompt_len:])
        assert len(full_ids) == len(labels), "full_ids and labels must have the same length"
        
        # 2. 截断
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            labels = labels[:max_seq_len]
        
        # 3. padding
        seq_len = len(full_ids)
        pad_len = max_seq_len - seq_len
        
        attention_mask = [1] * seq_len + [0] * pad_len
        token_ids = full_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [ignore_index] * pad_len
        
        # 4. position ids, 简单递增
        position_ids = list(range(max_seq_len))
        
        # 5. length， 记录 padding 前的 token id 长度
        length = seq_len
        
        processed_data.append({
            "token_ids": token_ids,
            "labels": labels,
            "position_ids": position_ids,
            "length": length,
            "attention_mask": attention_mask,
            "type": data_type
        })
        
        # 达到 batch_size 时写入磁盘并清空内存
        if len(processed_data) >= batch_size:
            writer = write_batch(processed_data, writer, schema)
            total_samples += len(processed_data)
            batch_count += 1
            processed_data = []  # 清空已处理的数据，释放内存
    
    # 写入剩余的数据
    if processed_data:
        writer = write_batch(processed_data, writer, schema)
        total_samples += len(processed_data)
        batch_count += 1
    
    # 关闭 writer
    if writer is not None:
        writer.close()
    
    print("-" * 30)
    print(f"SFT data merging and conversion completed!")
    print(f"Total samples: {total_samples:,}")
    print(f"Total batches: {batch_count}")
    print(f"Output file: {output_file}")
    print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print("-" * 30)


def analyze_sft_data(
    parquet_data_path: str = str(parquet_sft_data_path / "sft_data.parquet"),
    interval: int = 50,
    batch_size: int = 5000
    ):
    """
    分析 parquet 文件数据的长短分布情况, 并保存为条形图
    
    Args:
        parquet_data_path (str): parquet 文件路径
        interval (int): 统计区间大小, 默认为 50
        batch_size (int): 流式读取 parquet 时每批读取的行数, 默认为 5000
    """
    print(f"Analyzing SFT data length distribution: {parquet_data_path}")
    # 从 parquet 文件流式读取 length 列
    print("Reading parquet file in streaming mode...")
    parquet_file = pq.ParquetFile(parquet_data_path)
    total_rows = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None

    if 'length' not in parquet_file.schema_arrow.names:
        raise ValueError("The parquet file does not contain a 'length' column")

    total_samples = 0
    min_length = None
    max_length = None
    sum_length = 0
    length_distribution = defaultdict(int)

    progress_bar = tqdm(total=total_rows, desc="Analyzing length", unit="rows")
    for batch in parquet_file.iter_batches(columns=['length'], batch_size=batch_size):
        batch_lengths = batch.column(0).to_pylist()
        for length in batch_lengths:
            length = int(length)
            total_samples += 1
            sum_length += length

            if min_length is None or length < min_length:
                min_length = length
            if max_length is None or length > max_length:
                max_length = length

            bin_index = length // interval
            bin_start = bin_index * interval
            bin_end = bin_start + interval - 1
            bin_label = f"{bin_start}-{bin_end}"
            length_distribution[bin_label] += 1

        progress_bar.update(len(batch_lengths))
    progress_bar.close()

    if total_samples == 0:
        raise ValueError("The parquet file is empty")

    avg_length = float(sum_length / total_samples)
    
    print(f"Sequence Length Statistics:")
    print(f"  Minimum Length: {min_length}")
    print(f"  Maximum Length: {max_length}")
    print(f"  Average Length: {avg_length:.2f}")
    
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
        "total_samples": total_samples,
        "min_length": min_length,
        "max_length": max_length,
        "avg_length": avg_length,
        "length_distribution": dict(sorted_bins)
    }


def sample_from_sft_parquet(
    sft_parquet_path: str,
    sampled_sft_parquet_path: str,
    sample_num: dict,
    batch_size: int = 5000
    ):
    """
    从 sft_data.parquet 文件中按 type 字段进行抽样，生成 sampled_sft_data.parquet 文件
    
    Args:
        sft_parquet_path (str): 输入的 parquet 文件路径
        sampled_sft_parquet_path (str): 输出的抽样后 parquet 文件路径
        sample_num (dict): 每个 type 需要抽样的数量，例如 {"single_turn": 40000, "multi_turn": 10000, "self_cognition": 200}
        batch_size (int): 流式读取 parquet 时每批读取的行数, 默认为 5000

    Note:
        采用两遍流式处理提升效率：
        1) 第一遍只读取 type 列，用水库抽样得到各 type 的目标全局行号
        2) 第二遍读取全列，但只提取命中的行，避免把所有行都转为 Python dict
    """
    print("Starting to sample from SFT parquet file...")
    print(f"Input file: {sft_parquet_path}")
    print(f"Output file: {sampled_sft_parquet_path}")
    print(f"Sample numbers: {sample_num}")
    
    # 确保输出目录存在
    output_path = Path(sampled_sft_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 流式读取 parquet 文件，按 type 做水库抽样
    print("Reading parquet file in streaming mode...")
    parquet_file = pq.ParquetFile(sft_parquet_path)
    total_rows = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None

    # 检查是否有 type 字段
    if 'type' not in parquet_file.schema_arrow.names:
        raise ValueError("The parquet file does not contain a 'type' column")

    type_counts = defaultdict(int)
    sampled_index_by_type = {type_name: [] for type_name in sample_num}

    # 第一遍：只读 type 列，做每个 type 的水库抽样（保存全局行号）
    print("Pass 1/2: scanning type column and building reservoirs...")
    global_row_idx = 0
    progress_bar = tqdm(total=total_rows, desc="Pass 1/2 Sampling index", unit="rows")
    for batch in parquet_file.iter_batches(columns=['type'], batch_size=batch_size):
        type_values = batch.column(0).to_pylist()

        for local_idx, type_name in enumerate(type_values):
            if type_name is None:
                global_row_idx += 1
                continue

            type_counts[type_name] += 1

            if type_name in sample_num:
                target_num = sample_num[type_name]
                if target_num > 0:
                    bucket = sampled_index_by_type[type_name]
                    seen_count = type_counts[type_name]
                    current_global_idx = global_row_idx

                    if len(bucket) < target_num:
                        bucket.append(current_global_idx)
                    else:
                        replace_pos = random.randint(1, seen_count)
                        if replace_pos <= target_num:
                            bucket[replace_pos - 1] = current_global_idx

            global_row_idx += 1

        progress_bar.update(len(type_values))
    progress_bar.close()

    total_input_samples = sum(type_counts.values())
    print(f"Total samples in input file: {total_input_samples:,}")

    # 统计每个 type 的数量
    type_counts = dict(type_counts)
    print("\nType distribution in input file:")
    for type_name, count in sorted(type_counts.items()):
        print(f"  {type_name}: {count:,}")

    # 汇总各 type 抽样索引
    sampled_indices = []
    for type_name, target_num in sample_num.items():
        if type_name not in type_counts:
            print(f"Warning: Type '{type_name}' not found in the data, skipping...")
            continue

        available_count = type_counts[type_name]

        # 如果可用数量小于目标数量，使用全部数据
        if available_count <= target_num:
            print(f"  {type_name}: Using all {available_count:,} samples (requested: {target_num:,})")
            sampled_indices.extend(sampled_index_by_type[type_name])
        else:
            print(f"  {type_name}: Sampled {target_num:,} from {available_count:,} samples")
            sampled_indices.extend(sampled_index_by_type[type_name])

    # 第二遍：仅提取命中的行，避免全量 to_pylist
    print("Pass 2/2: loading selected rows only...")
    sampled_indices = sorted(sampled_indices)
    sampled_rows = []
    if sampled_indices:
        target_ptr = 0
        current_start_idx = 0
        progress_bar = tqdm(total=total_rows, desc="Pass 2/2 Collect rows", unit="rows")

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            rows_in_batch = batch.num_rows
            batch_end_idx = current_start_idx + rows_in_batch

            local_positions = []
            while target_ptr < len(sampled_indices) and sampled_indices[target_ptr] < batch_end_idx:
                local_positions.append(sampled_indices[target_ptr] - current_start_idx)
                target_ptr += 1

            if local_positions:
                table = pa.Table.from_batches([batch])
                selected_table = table.take(pa.array(local_positions, type=pa.int64()))
                sampled_rows.extend(selected_table.to_pylist())

            current_start_idx = batch_end_idx
            progress_bar.update(rows_in_batch)

            if target_ptr >= len(sampled_indices):
                break

        progress_bar.close()

    # 合并所有抽样结果
    if not sampled_rows:
        print("Error: No data sampled!")
        return

    # 打乱顺序
    random.shuffle(sampled_rows)

    # 保存到 parquet 文件
    sampled_table = pa.Table.from_pylist(sampled_rows)
    pq.write_table(sampled_table, sampled_sft_parquet_path)

    # 统计抽样后的分布
    sampled_type_counts = defaultdict(int)
    for row in sampled_rows:
        sampled_type_counts[row.get('type', 'unknown')] += 1
    
    print("\n" + "-" * 30)
    print("Sampling completed!")
    print(f"Total sampled samples: {len(sampled_rows):,}")
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
    ignore_index: int = -100,
    chunk_size: int = 20000,
    batch_size: int = 5000
    ):
    """
    对 sft_data.parquet 文件进行 packing, 将多个短序列打包到一个序列中, 提高训练效率
    
    采用 greedy packing 策略：
    1. 按长度从大到小排序
    2. 依次将样本放入 bin 中，尽量填满每个 bin
    3. 构建拼接后的 token_ids, labels, position_ids
    
    Args:
        sft_parquet_path (str): 输入的 parquet 文件路径
        packed_sft_parquet_path (str): 输出的 packing 后 parquet 文件路径
        max_seq_len (int): 最大序列长度, 默认为 512
        ignore_index (int): 用于 padding 的 ignore index, 默认为 -100
        chunk_size (int): 分片 packing 的样本数, 默认为 20000
        batch_size (int): 流式读取 parquet 时每批读取的行数, 默认为 5000
    """
    print("Starting to pack SFT parquet file...")
    print(f"Input file: {sft_parquet_path}")
    print(f"Output file: {packed_sft_parquet_path}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Chunk size: {chunk_size}")
    
    # 确保输出目录存在
    output_path = Path(packed_sft_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取 parquet 文件（流式）
    print("Reading parquet file in streaming mode...")
    parquet_file = pq.ParquetFile(sft_parquet_path)
    total_rows = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None

    # 检查必要的字段
    required_fields = ['token_ids', 'labels', 'length', 'type']
    for field in required_fields:
        if field not in parquet_file.schema_arrow.names:
            raise ValueError(f"The parquet file does not contain a '{field}' column")

    output_schema = pa.schema([
        ('token_ids', pa.list_(pa.int32())),
        ('labels', pa.list_(pa.int32())),
        ('position_ids', pa.list_(pa.int32())),
        ('length', pa.int32()),
        ('attention_mask', pa.list_(pa.int32())),
        ('type', pa.string()),
        ('num_samples', pa.int32()),
        ('sample_lengths', pa.list_(pa.int32()))
    ])

    def pack_samples_with_greedy(samples):
        """对一个分片内样本执行 greedy packing（按长度降序）"""
        if not samples:
            return []

        samples.sort(key=lambda x: x['length'], reverse=True)
        bins = []

        for sample in samples:
            placed = False
            for bin_item in bins:
                if bin_item['current_len'] + sample['length'] <= max_seq_len:
                    bin_item['samples'].append(sample)
                    bin_item['current_len'] += sample['length']
                    placed = True
                    break

            if not placed:
                bins.append({
                    'samples': [sample],
                    'current_len': sample['length']
                })

        return bins

    writer = None
    total_pad_token = 0
    total_bins = 0
    total_input_samples = 0
    total_samples_in_bins = 0

    def write_packed_chunk(samples_chunk, writer):
        nonlocal total_pad_token, total_bins, total_samples_in_bins

        bins = pack_samples_with_greedy(samples_chunk)
        packed_rows = []

        for bin_item in bins:
            bin_token_ids = []
            bin_labels = []
            position_ids = []
            bin_types = []

            for sample in bin_item['samples']:
                bin_token_ids.extend(sample['token_ids'])
                bin_labels.extend(sample['labels'])
                bin_types.append(sample['type'])
                position_ids.extend(list(range(sample['length'])))

            seq_len = len(bin_token_ids)
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                total_pad_token += pad_len
                bin_token_ids += [tokenizer.pad_token_id] * pad_len
                bin_labels += [ignore_index] * pad_len
                position_ids += [0] * pad_len

            attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
            sample_lengths = [sample['length'] for sample in bin_item['samples']]
            main_type = bin_types[0] if bin_types else "unknown"

            packed_rows.append({
                "token_ids": bin_token_ids,
                "labels": bin_labels,
                "position_ids": position_ids,
                "length": seq_len,
                "attention_mask": attention_mask,
                "type": main_type,
                "num_samples": len(bin_item['samples']),
                "sample_lengths": sample_lengths
            })

        if packed_rows:
            packed_table = pa.Table.from_pylist(packed_rows, schema=output_schema)
            if writer is None:
                writer = pq.ParquetWriter(packed_sft_parquet_path, output_schema)
            writer.write_table(packed_table)

            total_bins += len(packed_rows)
            total_samples_in_bins += sum(row['num_samples'] for row in packed_rows)

        return writer

    samples_buffer = []
    progress_bar = tqdm(total=total_rows, desc="Reading and preprocessing", unit="rows")
    for batch in parquet_file.iter_batches(columns=required_fields, batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        rows = table.to_pylist()

        for row in rows:
            token_ids = row['token_ids']
            labels = row['labels']
            length = int(row['length'])
            data_type = row['type']

            valid_token_ids = token_ids[:length]
            valid_labels = labels[:length]

            if length > max_seq_len:
                valid_token_ids = valid_token_ids[:max_seq_len]
                valid_labels = valid_labels[:max_seq_len]
                length = max_seq_len

            samples_buffer.append({
                "token_ids": valid_token_ids,
                "labels": valid_labels,
                "length": length,
                "type": data_type
            })
            total_input_samples += 1

            if len(samples_buffer) >= chunk_size:
                writer = write_packed_chunk(samples_buffer, writer)
                samples_buffer = []
        progress_bar.update(len(rows))
    progress_bar.close()

    if samples_buffer:
        writer = write_packed_chunk(samples_buffer, writer)

    if writer is not None:
        writer.close()

    if total_bins == 0:
        print("Error: No data packed!")
        return

    print(f"Packed {total_input_samples:,} samples into {total_bins:,} bins")

    # 统计信息
    pad_ratio = total_pad_token / (total_bins * max_seq_len)
    avg_samples_per_bin = total_samples_in_bins / total_bins
    
    print("\n" + "-" * 30)
    print("Packing completed!")
    print(f"Total bins: {total_bins:,}")
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
    max_seq_len = 2048
    
    # --------------------------------- 处理 sft 数据集 ---------------------------------
    # step 1. 处理 deepctrl 数据集
    # 默认筛选出所有符合条件的样本，保存为 jsonl 文件
    # process_sft_deepctrl(data_path=str(deepctrl_file_path), jsonl_path=str(deepctrl_jsonl_path), max_seq_len=max_seq_len)

    # step 2. 生成自我认知数据集
    # process_self_cognition(jsonl_path=str(self_cognition_jsonl_path), model_name="Mini-LLM", owner="WKQ", num_samples=300)

    # ------------------------------------- 合并多个数据集 ------------------------------------
    # step 3. 合并文件
    # 默认合并形成全部符合条件的样本，保存为 sft_data.parquet 文件，其中的样本已经过 tokenize 处理
    # 如果需调整 sft 训练量，仅需从 sft_data.parquet 文件中随机抽取样本即可
    # merge_and_convert_to_parquet(merge_list=["deepctrl.jsonl", "self_cognition.jsonl"], max_seq_len=max_seq_len)

    # ------------------------------------- 抽样函数 ------------------------------------
    # 可以选择从 sft_data.parquet 中按 type 进行抽样，形成 sampled_sft_data.parquet 文件
    # sample_from_sft_parquet(
    #     sft_parquet_path=str(parquet_sft_data_path / "sft_data.parquet"), 
    #     sampled_sft_parquet_path=str(parquet_sft_data_path / "sampled_sft_data.parquet"),
    #     sample_num = {"single_turn": 60000, "multi_turn": 20000, "self_cognition": 300}
    #     )

    # ------------------------------------- packing 函数 ------------------------------------
    # 形成经过 packing 的 parquet 文件，采用分片 packing 策略提高效率
    # pack_sft_parquet(
    #     sft_parquet_path=str(parquet_sft_data_path / "sampled_sft_data.parquet"),
    #     packed_sft_parquet_path=str(parquet_sft_data_path / "packed_sft_data.parquet"),
    #     max_seq_len=max_seq_len,
    #     )
    
    # ------------------------------------- 分析 sft 数据长度分布 ------------------------------------
    # 分析合并的、抽样的 sft 数据长度分布
    analyze_sft_data(parquet_data_path=str(parquet_sft_data_path / "sft_data.parquet"), interval=200)
    analyze_sft_data(parquet_data_path=str(parquet_sft_data_path / "sampled_sft_data.parquet"), interval=200)