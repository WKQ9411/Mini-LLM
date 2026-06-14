import json
import argparse
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
    AddedToken,
)
from transformers import AutoTokenizer
from pathlib import Path
import pandas as pd

from prepare_pretrain_data import analyze_and_sample_fineweb_edu, format_sample_percent, process_pretrain_fineweb_edu


# 定义路径
root_path = Path(__file__).parent.parent
# architecture_lab
arch_lab_path = root_path / "architecture_lab"
arch_lab_tokenizer_path = arch_lab_path / "backend" / "tokenizer"
arch_lab_data_path = arch_lab_path / "data"
# pretrain
pretrain_data_path = root_path / "data" / "pretrain_data"
# OpenCSG Fineweb-Edu-Chinese-V2.1 数据集
# https://www.modelscope.cn/datasets/opencsg/Fineweb-Edu-Chinese-V2.1
fineweb_edu_file_path = pretrain_data_path / "fineweb_edu"
# tokenizer 训练数据
tokenizer_data_path = root_path / "data" / "tokenizer_data"
# chat template
chat_template_file = tokenizer_data_path / "chat_template.jinja2"


# --------------------------------- 分层采样 FineWeb-Edu ---------------------------------
def sample_fineweb_edu(sample_ratio: float = 0.01) -> str:
    """
    按比例分层采样 FineWeb-Edu 数据，采样结果供 tokenizer 训练和训练数据分词共同使用

    Args:
        sample_ratio (float): 采样比例，默认 0.01 (1%)

    Returns:
        str: 采样数据的输出目录路径
    """
    print("=" * 30)
    print(f"Sampling FineWeb-Edu data (ratio={sample_ratio}) ...")
    print("=" * 30)

    sample_percent = format_sample_percent(sample_ratio)
    sampled_output_path = str(arch_lab_data_path / f"fineweb_edu_sampled_{sample_percent}_percent")

    analyze_and_sample_fineweb_edu(
        data_path=str(fineweb_edu_file_path),
        output_path=sampled_output_path,
        sample_ratio=sample_ratio,
    )

    print(f"Sampled data saved to: {sampled_output_path}")
    return sampled_output_path


# --------------------------------- 训练 architecture lab 专用 tokenizer ---------------------------------
def train_arch_lab_tokenizer(vocab_size: int = 10000, sampled_data_path: str = None, tokenizer_data_ratio: float = 0.1):
    """
    为 architecture lab 训练一个小词表的 BPE tokenizer

    Args:
        vocab_size (int): 基础词表大小 (不含额外添加的 token) 默认 10000
        sampled_data_path (str): 已采样数据的目录路径，作为 tokenizer 训练语料
        tokenizer_data_ratio (float): 用于 tokenizer 训练的数据占采样数据的比例，默认 0.1 (10%)
    """
    print("=" * 30)
    print(f"Training tokenizer with vocab_size={vocab_size}, tokenizer_data_ratio={tokenizer_data_ratio} ...")
    print("=" * 30)

    # tokenizer 训练语料目录（使用 sample_ratio 采样的数据）
    data_file = Path(sampled_data_path)
    save_path = arch_lab_tokenizer_path
    save_path.mkdir(parents=True, exist_ok=True)

    # 初始化 tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 定义特殊 token
    special_tokens = [""]  # pad token，会被分配 id 0

    # 设置训练器并添加特殊 token
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=3,
    )

    # 读取语料文件（按 tokenizer_data_ratio 比例抽取，减少内存占用）
    def read_data(data_file, ratio):
        for parquet_file in data_file.glob("**/*.parquet"):
            df = pd.read_parquet(parquet_file)
            if ratio < 1.0:
                df = df.sample(frac=ratio, random_state=42)
            for _, row in df.iterrows():
                yield row['text']

    # 训练 tokenizer
    tokenizer.train_from_iterator(
        iterator=read_data(data_file, tokenizer_data_ratio),
        trainer=trainer,
    )

    # 添加额外 token（与 train_tokenizer.py 保持一致）
    added_content = ["<|im_start|>", "<|im_end|>", "breadcrumbs", ""]
    special_flags = [True, True, False, False]

    added_tokens = [
        AddedToken(content,
                   single_word=False, lstrip=False, rstrip=False,
                   normalized=False, special=sp)
        for content, sp in zip(added_content, special_flags)
    ]

    num_added_tokens = tokenizer.add_tokens(added_tokens)
    print(f"{num_added_tokens} new tokens added to the tokenizer.")
    print(f"Total vocab size: {tokenizer.get_vocab_size()}")

    # 保存 tokenizer
    tokenizer.save(str(save_path / "tokenizer.json"))
    tokenizer.model.save(str(save_path))

    # 读取 chat template
    with open(str(chat_template_file), 'r', encoding='utf-8') as f:
        chat_template = f.read()
    tokenizer.chat_template = chat_template

    # 创建配置文件
    total_vocab_size = tokenizer.get_vocab_size()
    config = {
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            str(vocab_size): {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            str(vocab_size + 1): {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            str(vocab_size + 2): {
                "content": "breadcrumbs",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": False,
            },
            str(vocab_size + 3): {
                "content": "",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": False,
            },
        },
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>",
        ],
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "pad_token": "",
        "unk_token": None,
        "chat_template": chat_template,
        "clean_up_tokenization_spaces": False,
        "errors": "replace",
        "model_max_length": 10000000,
        "split_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "add_bos_token": False,
    }

    with open(str(save_path / "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print(f"Tokenizer saved to: {save_path}")
    print(f"Total vocab size (with added tokens): {total_vocab_size}")
    print("Tokenizer training completed!")


# --------------------------------- 分词 + 保存 .bin ---------------------------------
def tokenize_and_save(sampled_data_path: str, sample_ratio: float = 0.01):
    """
    用 architecture lab 的 tokenizer 对采样数据分词并保存为 .bin 文件

    Args:
        sampled_data_path (str): 已采样数据的目录路径
        sample_ratio (float): 采样比例，仅用于打印信息
    """
    print("=" * 30)
    print(f"Tokenizing sampled data (ratio={sample_ratio}) ...")
    print("=" * 30)

    # 加载 architecture lab 的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(arch_lab_tokenizer_path))
    eos_token = tokenizer.eos_token
    print(f"Loaded tokenizer from: {arch_lab_tokenizer_path}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # 分词并保存为 .bin
    bin_output_path = str(arch_lab_data_path / "train.bin")

    # 替换 prepare_pretrain_data 模块中的全局 tokenizer，使其使用 architecture lab 的 tokenizer
    import prepare_pretrain_data
    prepare_pretrain_data.tokenizer = tokenizer
    prepare_pretrain_data.eos_token = eos_token

    process_pretrain_fineweb_edu(
        data_path=sampled_data_path,
        bin_path=bin_output_path,
    )

    print("=" * 30)
    print("Architecture lab data preparation completed!")
    print(f"Sample ratio: {sample_ratio}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Binary data saved to: {bin_output_path}")
    print("=" * 30)


# --------------------------------- 主入口 ---------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare architecture lab training data")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Tokenizer vocabulary size (default: 10000)")
    parser.add_argument("--sample_ratio", type=float, default=0.01, help="Sampling ratio from FineWeb-Edu (default: 0.01)")
    parser.add_argument("--skip_tokenizer", action="store_true", help="Skip tokenizer training, use existing tokenizer in architecture_lab/backend/tokenizer/")
    parser.add_argument("--skip_sampling", action="store_true", help="Skip sampling, use existing sampled data (path is inferred from --sample_ratio)")
    parser.add_argument("--tokenizer_data_ratio", type=float, default=0.1, help="Ratio of sampled data used for tokenizer training (default: 0.1, lower saves memory)")

    args = parser.parse_args()

    # step 1: 分层采样（可选跳过，直接使用已有的采样数据）
    if not args.skip_sampling:
        sampled_data_path = sample_fineweb_edu(sample_ratio=args.sample_ratio)
    else:
        sample_percent = format_sample_percent(args.sample_ratio)
        sampled_data_path = str(arch_lab_data_path / f"fineweb_edu_sampled_{sample_percent}_percent")
        print(f"Skipping sampling (using existing sampled data): {sampled_data_path}")

    # step 2: 训练 tokenizer（可选跳过，使用采样数据作为训练语料）
    if not args.skip_tokenizer:
        train_arch_lab_tokenizer(vocab_size=args.vocab_size, sampled_data_path=sampled_data_path, tokenizer_data_ratio=args.tokenizer_data_ratio)
    else:
        print("Skipping tokenizer training (using existing tokenizer)")

    # step 3: 分词 + 保存 .bin（复用 step 1 的采样数据）
    tokenize_and_save(sampled_data_path=sampled_data_path, sample_ratio=args.sample_ratio)
