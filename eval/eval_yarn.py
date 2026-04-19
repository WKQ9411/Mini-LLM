import argparse
import math
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import mini_models


root_path = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YaRN PPL Curve")

    # 模型
    parser.add_argument("--base_model_path", type=str, default=str(root_path / "output/pretrained_mini_llama3"), help="Path to the pretrained (base) model")
    parser.add_argument("--yarn_finetuned_model_path", type=str, default=str(root_path / "output/yarn_mini_llama3"), help="Path to the YaRN finetuned model")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Model dtype")

    # 数据
    # 当前是从 fineweb_edu 数据集中采样一些长序列数据来进行评估，因此需至少保证下载一个 parquet 文件
    parser.add_argument("--fineweb_edu_path", type=str, default=str(root_path / "data/pretrain_data/fineweb_edu"), help="Path to fineweb_edu parquet directory")
    parser.add_argument("--sample_count", type=int, default=64, help="Number of evaluation samples (all samples are filtered to length > max_seq_len)")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for evaluation")
    parser.add_argument("--step", type=int, default=256, help="Step size for sequence length, starts from step until max_seq_len")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_scan_rows", type=int, default=600000, help="Max rows to scan before early stop if enough long samples are found")

    # YaRN
    parser.add_argument("--yarn_alpha", type=float, default=1.0, help="YaRN alpha (beta_slow)")
    parser.add_argument("--yarn_beta", type=float, default=32.0, help="YaRN beta (beta_fast)")

    # 画图
    parser.add_argument("--output_figure_path", type=str, default=str(root_path / "assets" / "eval_yarn_ppl_curve.png"), help="Output path for PPL curve figure")
    parser.add_argument("--y_log_scale", action="store_true", help="Use log scale on y-axis for better visual separation")

    return parser.parse_args()


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    return torch.float32


def get_text_column(df: pd.DataFrame) -> str:
    candidate_columns = ["text", "content", "raw_content", "document"]
    for col in candidate_columns:
        if col in df.columns:
            return col

    string_columns = [col for col in df.columns if df[col].dtype == "object"]
    if not string_columns:
        raise ValueError("No text-like column found in parquet file")
    return string_columns[0]


def collect_long_samples(
    parquet_root: Path,
    tokenizer,
    sample_count: int,
    max_seq_len: int,
    seed: int,
    max_scan_rows: int,
    ) -> List[List[int]]:
    parquet_files = list(parquet_root.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {parquet_root}")

    rng = random.Random(seed)
    rng.shuffle(parquet_files)

    eos = tokenizer.eos_token
    samples: List[List[int]] = []
    scanned_rows = 0

    print(f"Found {len(parquet_files)} parquet files, start sampling long sequences...")
    for parquet_file in parquet_files:
        if len(samples) >= sample_count:
            break
        if scanned_rows >= max_scan_rows:
            print(f"Reach max_scan_rows={max_scan_rows}, stop scanning early")
            break

        df = pd.read_parquet(parquet_file)
        if len(df) == 0:
            continue

        text_col = get_text_column(df)
        row_indices = list(range(len(df)))
        rng.shuffle(row_indices)

        for idx in row_indices:
            if len(samples) >= sample_count:
                break
            if scanned_rows >= max_scan_rows:
                break

            scanned_rows += 1
            text = df.iloc[idx][text_col]
            if not isinstance(text, str) or not text.strip():
                continue

            token_ids = tokenizer.encode(text + eos, add_special_tokens=False)
            if len(token_ids) <= max_seq_len:
                continue

            samples.append(token_ids[:max_seq_len])

    if not samples:
        raise ValueError(
            f"No valid sample found (token length > {max_seq_len}). "
            f"Please check dataset path: {parquet_root}"
        )

    if len(samples) < sample_count:
        print(f"Warning: only collected {len(samples)} samples (target {sample_count})")

    print(f"Collected {len(samples)} long samples for evaluation")
    return samples


def build_yarn_config(config, target_max_seq_len: int, yarn_alpha: float, yarn_beta: float):
    original_max_position_embeddings = int(config.max_position_embeddings)
    factor = target_max_seq_len / original_max_position_embeddings
    rope_scaling = {
        "rope_type": "yarn",
        "factor": factor,
        "attention_factor": None,
        "beta_fast": yarn_beta,
        "beta_slow": yarn_alpha,
    }
    config.max_position_embeddings = target_max_seq_len
    config.rope_scaling = rope_scaling
    return config


def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    inject_yarn: bool = False,
    target_max_seq_len: int = 2048,
    yarn_alpha: float = 1.0,
    yarn_beta: float = 32.0,
    ):
    config = AutoConfig.from_pretrained(model_path)

    if inject_yarn:
        config = build_yarn_config(
            config=config,
            target_max_seq_len=target_max_seq_len,
            yarn_alpha=yarn_alpha,
            yarn_beta=yarn_beta,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=dtype)

    model = model.to(device)
    model.eval()
    return model


def evaluate_ppl_curve(
    model,
    samples: List[List[int]],
    lengths: List[int],
    batch_size: int,
    device: torch.device,
    enable_amp: bool,
    amp_dtype: torch.dtype,
    ) -> List[float]:
    ppl_values = []

    print(f"Evaluating PPL curve for lengths: {lengths} with batch size {batch_size}...")
    for i, cur_len in enumerate(lengths):
        batched_losses = []
        for start in range(0, len(samples), batch_size):
            batch_tokens = samples[start : start + batch_size]
            input_ids = torch.tensor(
                [token_ids[:cur_len] for token_ids in batch_tokens],
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                if enable_amp:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                else:
                    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)

            batched_losses.append(outputs.loss.item())

        mean_loss = sum(batched_losses) / len(batched_losses)
        ppl = math.exp(min(mean_loss, 20.0))  # 设置一个 loss 上限
        ppl_values.append(ppl)
        print(f"step: {i+1}/{len(lengths)}, length={cur_len}, loss={mean_loss:.6f}, ppl={ppl:.6f}")

    return ppl_values


def plot_ppl_curve(
    lengths: List[int],
    base_ppl: List[float],
    yarn_ft_ppl: List[float],
    base_with_yarn_ppl: List[float],
    save_path: Path,
    y_log_scale: bool = False,
    ):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    color_base = "tab:blue"
    color_yarn_ft = "tab:orange"
    color_yarn_no_ft = "tab:green"

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.1, 1.4]})

    ax_main.plot(lengths, base_ppl, marker="o", linewidth=2, color=color_base, label="Pretrained (Original)")
    ax_main.plot(lengths, yarn_ft_ppl, marker="s", linewidth=2, color=color_yarn_ft, label="YaRN Finetuned")
    ax_main.plot(
        lengths,
        base_with_yarn_ppl,
        marker="^",
        linewidth=2,
        color=color_yarn_no_ft,
        label="Pretrained + YaRN Params (No FT)",
    )
    ax_main.set_xlabel("Sequence Length")
    ax_main.set_ylabel("Perplexity (PPL)")
    ax_main.set_title("YaRN PPL Comparison")
    if y_log_scale:
        tiny = 1e-8
        ax_main.set_yscale("log")
        ax_main.set_ylim(
            max(min(min(base_ppl), min(yarn_ft_ppl), min(base_with_yarn_ppl)), tiny),
            max(max(base_ppl), max(yarn_ft_ppl), max(base_with_yarn_ppl)),
        )
    ax_main.grid(True, linestyle="--", alpha=0.4)
    ax_main.legend()

    # 放大显示 YaRN 微调 vs 仅注入 YaRN 参数（不微调）的差异
    ax_zoom.plot(lengths, yarn_ft_ppl, marker="s", linewidth=2, color=color_yarn_ft, label="YaRN Finetuned")
    ax_zoom.plot(lengths, base_with_yarn_ppl, marker="^", linewidth=2, color=color_yarn_no_ft, label="+YaRN Params (No FT)")
    ax_zoom.fill_between(lengths, yarn_ft_ppl, base_with_yarn_ppl, alpha=0.2, color=color_yarn_no_ft, label="Gap")

    yarn_pair_min = min(min(yarn_ft_ppl), min(base_with_yarn_ppl))
    yarn_pair_max = max(max(yarn_ft_ppl), max(base_with_yarn_ppl))
    yarn_pair_span = max(yarn_pair_max - yarn_pair_min, 1e-8)
    y_margin = max(yarn_pair_span * 0.15, 1e-4)
    ax_zoom.set_ylim(yarn_pair_min - y_margin, yarn_pair_max + y_margin)

    delta_abs = [no_ft - ft for ft, no_ft in zip(yarn_ft_ppl, base_with_yarn_ppl)]
    delta_rel = [((no_ft - ft) / max(no_ft, 1e-8)) * 100.0 for ft, no_ft in zip(yarn_ft_ppl, base_with_yarn_ppl)]
    mean_delta_abs = sum(delta_abs) / len(delta_abs)
    mean_delta_rel = sum(delta_rel) / len(delta_rel)

    ax_zoom.set_xlabel("Sequence Length")
    ax_zoom.set_ylabel("Perplexity (PPL)")
    ax_zoom.set_title("Zoom: YaRN FT vs No-FT")
    if y_log_scale:
        tiny = 1e-8
        ax_zoom.set_yscale("log")
        ax_zoom.set_ylim(max(yarn_pair_min - y_margin, tiny), max(yarn_pair_max + y_margin, tiny * 10))
    ax_zoom.grid(True, linestyle="--", alpha=0.4)
    ax_zoom.legend(loc="upper left")
    ax_zoom.text(
        0.02,
        0.02,
        f"Avg ΔPPL (NoFT-FT): {mean_delta_abs:.4f}\nAvg Rel Gain: {mean_delta_rel:.2f}%",
        transform=ax_zoom.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle("YaRN Evaluation: Overall Trend + Local Difference", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"PPL curve saved to: {save_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    enable_amp = device.type == "cuda" and dtype in [torch.float16, torch.bfloat16]
    amp_dtype = dtype

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))

    lengths = list(range(args.step, args.max_seq_len + 1, args.step))
    if lengths[-1] != args.max_seq_len:
        raise ValueError("max_seq_len must be divisible by step")

    # 获取评估样本
    samples = collect_long_samples(
        parquet_root=Path(args.fineweb_edu_path),
        tokenizer=tokenizer,
        sample_count=args.sample_count,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        max_scan_rows=args.max_scan_rows,
    )

    # step 1: 对预训练模型进行评估
    print("\nLoading model 1/3: Pretrained (Original)")
    base_model = load_model(
        model_path=args.base_model_path,
        device=device,
        dtype=dtype,
        inject_yarn=False,
    )
    base_ppl = evaluate_ppl_curve(
        model=base_model,
        samples=samples,
        lengths=lengths,
        batch_size=args.batch_size,
        device=device,
        enable_amp=enable_amp,
        amp_dtype=amp_dtype,
    )
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # step 2: 对加入 YaRN 参数后，且经过微调的模型进行评估
    print("\nLoading model 2/3: YaRN Finetuned")
    yarn_ft_model = load_model(
        model_path=args.yarn_finetuned_model_path,
        device=device,
        dtype=dtype,
        inject_yarn=False,
    )
    yarn_ft_ppl = evaluate_ppl_curve(
        model=yarn_ft_model,
        samples=samples,
        lengths=lengths,
        batch_size=args.batch_size,
        device=device,
        enable_amp=enable_amp,
        amp_dtype=amp_dtype,
    )
    del yarn_ft_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # step 3: 对只加入 YaRN 参数，不微调的模型进行评估
    print("\nLoading model 3/3: Pretrained + YaRN params (No FT)")
    base_with_yarn_model = load_model(
        model_path=args.base_model_path,
        device=device,
        dtype=dtype,
        inject_yarn=True,
        target_max_seq_len=args.max_seq_len,
        yarn_alpha=args.yarn_alpha,
        yarn_beta=args.yarn_beta,
    )
    base_with_yarn_ppl = evaluate_ppl_curve(
        model=base_with_yarn_model,
        samples=samples,
        lengths=lengths,
        batch_size=args.batch_size,
        device=device,
        enable_amp=enable_amp,
        amp_dtype=amp_dtype,
    )
    del base_with_yarn_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    plot_ppl_curve(
        lengths=lengths,
        base_ppl=base_ppl,
        yarn_ft_ppl=yarn_ft_ppl,
        base_with_yarn_ppl=base_with_yarn_ppl,
        save_path=Path(args.output_figure_path),
        y_log_scale=args.y_log_scale,
    )

    print("\nEvaluation summary:")
    print(f"lengths: {lengths}")
    print(f"pretrained ppl: {base_ppl}")
    print(f"yarn finetuned ppl: {yarn_ft_ppl}")
    print(f"pretrained + yarn params ppl: {base_with_yarn_ppl}")


if __name__ == "__main__":
    main()
