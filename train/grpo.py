import argparse
from collections import Counter
import datetime
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from data_loader import PreTrainDataset, GRPODataset
from mini_models import get_model_and_config, get_model_info, list_models
from utils import (
    configure_optimizer,
    create_folder,
    get_lr,
    load_args,
    plot_curve_grpo,
    save_args,
)


root_path = Path(__file__).parent.parent
support_models = ", ".join(list_models())


# -------------------------------------------【参数解析】------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Mini-LLM")

    # 模型与训练精度
    parser.add_argument("--model_name", type=str, required=True, help=f"Mini model names, support: {support_models}")
    parser.add_argument("--precision", type=str, default="bf16", help="Mixed precision training: default bf16, options are fp32 or fp16")
    parser.add_argument("--base_model_path", type=str, default=None, help="Base model checkpoint path, default: output/sft_{model_name}")

    # 共用超参
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum sequence length. If not set, will be inferred from model config.")
    parser.add_argument("--max_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="Beta parameters for AdamW optimizer")

    # [可选] Mid-training 参数（在任务领域上继续预训练）
    parser.add_argument("--mid_training", action="store_true", help="Enable mid-training (continued pretraining on domain data)")
    parser.add_argument("--mid_training_data_path", type=str, default=f"{root_path}/data/grpo_data/grpo.bin", help="Path to mid-training binary data")
    parser.add_argument("--mid_epochs", type=int, default=1, help="Mid-training epochs")
    parser.add_argument("--mid_max_lr", type=float, default=1e-5, help="Mid-training max learning rate")
    parser.add_argument("--mid_min_lr", type=float, default=1e-6, help="Mid-training min learning rate")
    parser.add_argument("--mid_warmup_ratio", type=float, default=0.05, help="Mid-training warmup ratio")
    parser.add_argument("--mid_lr_decay_ratio", type=float, default=0.98, help="Mid-training LR decay ratio")
    parser.add_argument("--mid_weight_decay", type=float, default=0.01, help="Mid-training weight decay")
    parser.add_argument("--mid_batch_size", type=int, default=None, help="Mid-training batch size (default: same as max_batch_size)")

    # [可选] 冷启动 SFT 参数（在 GRPO 前先微调，让模型学会基本的输出格式）
    parser.add_argument("--cold_start_sft", action="store_true", help="Enable cold-start SFT before GRPO training")
    parser.add_argument("--sft_epochs", type=int, default=1, help="Cold-start SFT epochs")
    parser.add_argument("--sft_max_lr", type=float, default=1e-5, help="Cold-start SFT max learning rate")
    parser.add_argument("--sft_min_lr", type=float, default=1e-6, help="Cold-start SFT min learning rate")
    parser.add_argument("--sft_warmup_ratio", type=float, default=0.05, help="Cold-start SFT warmup ratio")
    parser.add_argument("--sft_lr_decay_ratio", type=float, default=0.98, help="Cold-start SFT LR decay ratio")
    parser.add_argument("--sft_weight_decay", type=float, default=0.1, help="Cold-start SFT weight decay")
    parser.add_argument("--sft_max_grad_norm", type=float, default=1.0, help="Cold-start SFT max gradient norm")
    parser.add_argument("--sft_batch_size", type=int, default=None, help="Cold-start SFT batch size (default: same as max_batch_size)")
    parser.add_argument("--sft_data_path", type=str, default=f"{root_path}/data/grpo_data/cold_start.jsonl", help="Path to cold-start SFT jsonl dataset")

    # GRPO 训练参数
    parser.add_argument("--grpo_epochs", type=int, default=1, help="GRPO training epochs")
    parser.add_argument("--grpo_max_lr", type=float, default=1e-5, help="GRPO max learning rate")
    parser.add_argument("--grpo_min_lr", type=float, default=1e-6, help="GRPO min learning rate")
    parser.add_argument("--grpo_warmup_ratio", type=float, default=0.05, help="GRPO warmup ratio")
    parser.add_argument("--grpo_lr_decay_ratio", type=float, default=0.98, help="GRPO LR decay ratio")
    parser.add_argument("--grpo_weight_decay", type=float, default=0.01, help="GRPO weight decay")
    parser.add_argument("--group_size", type=int, default=4, help="Number of completions per prompt (G)")
    parser.add_argument("--grpo_clip_eps", type=float, default=0.2, help="PPO-style clipping epsilon")
    parser.add_argument("--kl_coeff", type=float, default=0.1, help="KL penalty coefficient against reference model")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate per completion")
    parser.add_argument("--temperature", type=float, default=1.2, help="Temperature for generation during GRPO")
    parser.add_argument("--no_grpo", action="store_true", help="Skip GRPO training, only run mid-training and/or cold-start SFT")

    # 路径与日志设置
    parser.add_argument("--grpo_data_path", type=str, default=f"{root_path}/data/grpo_data/grpo.jsonl", help="Path to GRPO jsonl dataset")
    parser.add_argument("--tokenizer_path", type=str, default=f"{root_path}/mini_tokenizer", help="Tokenizer path")
    parser.add_argument("--output_path", type=str, default=f"{root_path}/output", help="Model output directory")
    parser.add_argument("--log_interval", type=int, default=50, help="Training log print interval")

    args = parser.parse_args()
    return args


# -------------------------------------------【辅助函数】------------------------------------------- #
# 解析模型路径
def _resolve_model_paths(args):
    if args.base_model_path is None:
        args.base_model_path = str(root_path / f"output/sft_{args.model_name}")
    if not os.path.exists(args.base_model_path):
        raise FileNotFoundError(f"Base model path not found: {args.base_model_path}")


# 解析最大序列长度
def _resolve_max_seq_len(args) -> int:
    if args.max_seq_len is not None:
        max_seq_len = int(args.max_seq_len)
        print(f"Using user-specified max_seq_len={max_seq_len}")
        return max_seq_len

    training_args_files = sorted(Path(args.base_model_path).glob("*_training_args.json"))
    if training_args_files:
        training_args_path = str(training_args_files[0])
        base_model_training_args = load_args(training_args_path)
        max_seq_len = int(base_model_training_args["max_seq_len"])
        print(f"Loaded max_seq_len={max_seq_len} from: {training_args_path}")
        return max_seq_len

    _, Config = get_model_and_config(args.model_name)
    config = Config.from_pretrained(args.base_model_path)
    max_seq_len = int(config.max_position_embeddings)
    print(f"Loaded max_seq_len={max_seq_len} from config: {args.base_model_path}")
    return max_seq_len


# 特定模型参数调整
def _adjust_model_for_grpo(model, model_name: str) -> None:
    if model_name == "mini_deepseekv3":
        model.config.use_mtp = False
        model.config.use_noaux_load_balance = False
        model.config.use_seq_aux = False


# 加载模型并设置训练状态
def _load_model(
    model_name: str,
    model_path: str,
    device: torch.device,
    trainable: bool,
):
    Model, _ = get_model_and_config(model_name)
    model = Model.from_pretrained(model_path).to(device)
    _adjust_model_for_grpo(model, model_name)

    if trainable:
        model.train()
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


# 从生成文本中提取结构化内容
def _extract_json_from_response(text: str) -> str | None:
    """从 response 中提取 ```json ... ``` 包裹的 JSON 字符串"""
    matches = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if len(matches) == 1:
        return matches[0].strip()
    return None


# 解析预填写 <think>\n 后的模型生成结果
def _parse_prefilled_think_response(response_text: str) -> tuple[str | None, str | None]:
    """
    解析预填写 <think>\n 后的模型生成结果

    生成时 <think>\n 已经在 prompt 模板中, 因此模型只需要生成:
    1. think 内容
    2. </think>
    3. 最终答案
    """
    if response_text.count("<think>") != 0:
        return None, None
    if response_text.count("</think>") != 1:
        return None, None

    think_content, answer_text = response_text.split("</think>", maxsplit=1)
    return think_content.strip(), answer_text.lstrip()


def _measure_effective_think_length(think_content: str) -> int:
    """
    计算 think 的有效长度, 这里不统计空格、换行、制表符等空白字符, 避免模型通过灌水式空白膨胀来骗取长度奖励
    """
    return len(re.sub(r"\s+", "", think_content))


def _compute_char_ngram_repeat_ratio(think_content: str, n: int = 3) -> float:
    """
    计算 think 内容的字符级 n-gram 重复率

    先移除空白字符, 再统计重复 n-gram 在所有 n-gram 中的占比, 用于识别模板刷写、无意义重复字符等低质量 think
    """
    compact_text = re.sub(r"\s+", "", think_content)
    if len(compact_text) < n:
        return 0.0

    ngrams = [compact_text[i:i + n] for i in range(len(compact_text) - n + 1)]
    total_ngrams = len(ngrams)
    if total_ngrams == 0:
        return 0.0

    counts = Counter(ngrams)
    repeated_ngrams = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_ngrams / total_ngrams


def _compute_reward(
    response_text: str,
    ground_truth_response: str,
    think_min_len: int = 50,
    think_max_len: int = 150,
) -> tuple[float, dict[str, float]]:
    """
    计算总奖励和各分项详情

    奖励组成 (总分 0~1):
    1. 格式奖励 (0.1): 正确闭合 </think>, 且其后首个非空白内容必须是一个完整的 ```json...``` 块
    2. 思维链长度奖励 (0.1): 仅当模型严格进入 JSON 输出区后, 才考虑思维链长度奖励
       并且简单要求 think 的非空白有效长度在合理范围内, 3-gram 重复率不能过高, 防止偷懒堆思维链
    3. JSON 可解析奖励 (0.3): 严格位于 </think> 之后的 JSON 能被 json.loads 正确解析
    4. 答案一致性奖励 (0.5): 解析后的 JSON 与标准答案一致
    """
    details = {
        "format": 0.0,
        "think_length": 0.0,
        "parseable": 0.0,
        "correctness": 0.0,
    }

    # --- 1. 格式奖励: 预填写 `<think>\n` 后，检查是否正确闭合 think 并输出 JSON ---
    think_content, answer_text = _parse_prefilled_think_response(response_text)
    has_think = think_content is not None
    stripped_answer = (answer_text or "").lstrip()
    answer_starts_json = stripped_answer.startswith("```json")  # 正式输出，去除空白后要严格匹配 ```json
    json_matches = re.findall(r"```json\s*(.*?)\s*```", answer_text or "", re.DOTALL)
    has_json_block = len(json_matches) == 1  # json block 严格为 1 个

    if has_think and answer_starts_json and has_json_block:
        details["format"] = 0.1

    # --- 2. 思维链长度奖励 ---
    if has_think and answer_starts_json and has_json_block:
        think_len = _measure_effective_think_length(think_content)
        # 计算部分分的宽松边界，过短或过长不给分
        soft_min = think_min_len // 2
        soft_max = think_max_len + (think_max_len - think_min_len) * 3 // 4
        if think_min_len <= think_len <= think_max_len:
            details["think_length"] = 0.1  # 合理范围, 满分
        elif soft_min <= think_len < think_min_len or think_max_len < think_len <= soft_max:
            details["think_length"] = 0.05  # 稍短或稍长, 部分分

        if details["think_length"] > 0.0:
            repeat_ratio = _compute_char_ngram_repeat_ratio(think_content, n=3)
            # 重复率过高通常意味着在刷模板或无意义字符，对长度奖励做衰减。
            if repeat_ratio > 0.4:
                details["think_length"] = 0.0
            elif repeat_ratio > 0.25:
                details["think_length"] *= 0.5

    # --- 3. JSON 可解析奖励 ---
    # 只奖励正确闭合 think 之后的答案，避免模型直接在 think 块里输出 JSON
    if has_think and answer_starts_json and has_json_block:
        json_str = _extract_json_from_response(answer_text)
        parsed_output = None
        if json_str is not None:
            try:
                parsed_output = json.loads(json_str)
                details["parseable"] = 0.3
            except (json.JSONDecodeError, ValueError):
                pass

        # --- 4. 答案一致性奖励: 与标准答案对比 ---
        if parsed_output is not None:
            gt_json_str = _extract_json_from_response(ground_truth_response)
            if gt_json_str is not None:
                try:
                    gt_parsed = json.loads(gt_json_str)
                    if parsed_output == gt_parsed:
                        details["correctness"] = 0.5  # 完全一致
                    elif isinstance(parsed_output, dict) and isinstance(gt_parsed, dict):
                        # 部分匹配: 计算键值对重合比例
                        if gt_parsed:
                            matching_keys = sum(
                                1 for k in gt_parsed
                                if k in parsed_output and parsed_output[k] == gt_parsed[k]
                            )
                            overlap_ratio = matching_keys / len(gt_parsed)
                            details["correctness"] = 0.5 * overlap_ratio
                except (json.JSONDecodeError, ValueError):
                    pass

    total_reward = sum(details.values())
    return total_reward, details


# 计算序列的总 logprob 和有效 token 数量
def _compute_sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float()  # (batch_size, seq_len-1, vocab_size)
    shift_labels = labels[:, 1:]  # (batch_size, seq_len-1)
    loss_mask = shift_labels.ne(-100)  # ne 相当于 not equal，生成一个布尔掩码，标记哪些位置的标签不是 -100

    safe_labels = shift_labels.masked_fill(~loss_mask, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)  # (batch_size, seq_len-1)
    token_logps = token_logps * loss_mask

    seq_logps = token_logps.sum(dim=-1)  # (batch_size,)
    token_counts = loss_mask.sum(dim=-1)  # (batch_size,)
    return seq_logps, token_counts


# 计算 response 部分的归一化 logprob
def _compute_response_logps(
    model,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """前向计算并返回 response 部分的长度归一化 logprob"""
    outputs = model(
        input_ids=full_ids.to(device),
        attention_mask=attention_mask.to(device),
    )
    seq_logps, token_counts = _compute_sequence_logps(outputs.logits, labels.to(device))
    return seq_logps / token_counts.clamp_min(1).to(seq_logps.dtype)  # (batch_size,)


# 整 batch 生成 group_size 个补全
@torch.no_grad()
def _generate_completions(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    prompt_lengths: list[int],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    enable_amp: bool,
    autocast_dtype: torch.dtype | None,
) -> dict:
    """
    将 prompt 展平为 (B*G) 后整 batch 生成补全, 收集 old_logps 用于后续 GRPO 更新

    Returns:
        dict 包含:
        - full_ids: (B*G, max_total_len) 完整序列
        - attention_mask: (B*G, max_total_len)
        - labels: (B*G, max_total_len) prompt 部分为 -100
        - old_logps: (B*G,) 生成时收集的 logprob
        - response_texts: list[list[str]] 每个 prompt 的 G 个 response 文本
    """
    device = prompt_ids.device
    batch_size = prompt_ids.size(0)

    model.eval()
    expanded_batch_size = batch_size * group_size
    expanded_prompt_lengths = torch.tensor(prompt_lengths, device=device, dtype=torch.long).repeat_interleave(group_size)  # (B*G,)
    expanded_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)  # (B*G, max_prompt_len)
    expanded_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)  # (B*G, max_prompt_len)

    generated_tokens = []
    generated_logps = []
    generated_token_masks = []
    finished = torch.zeros(expanded_batch_size, dtype=torch.bool, device=device)  # 用于标记每个样本是否生成完毕 (B*G,)

    prefill_input_ids = expanded_prompt_ids[:, :-1]  # 用于 prefill 阶段，左填充，形状为 (B*G, max_prompt_len-1)
    prefill_attention_mask = expanded_prompt_mask[:, :-1]  # (B*G, max_prompt_len-1)
    prefill_position_ids = prefill_attention_mask.cumsum(dim=1) - 1  # 根据 mask 计算 position_ids (B*G, max_prompt_len-1)
    prefill_position_ids = prefill_position_ids.clamp_min(0)

    # ------- step 1. rollout ---------
    # prefill
    with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):
        outputs = model(
            input_ids=prefill_input_ids,
            attention_mask=prefill_attention_mask,
            position_ids=prefill_position_ids,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    cur_token = expanded_prompt_ids[:, [-1]]  # (B*G, 1)
    current_attention_mask = expanded_prompt_mask.clone()
    current_position_ids = (expanded_prompt_lengths - 1).unsqueeze(-1)  # 当前 token 的 position_ids (B*G, 1)

    # decode
    for _ in range(max_new_tokens):
        with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):
            outputs = model(
                input_ids=cur_token,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            next_logits = outputs.logits[:, -1, :].float()  # (B*G, vocab_size)
        past_key_values = outputs.past_key_values

        # 采样生成下一个 token
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)  # (B*G, vocab_size)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B*G,)
        else:
            next_tokens = next_logits.argmax(dim=-1)  # (B*G,)

        # 计算 old_logps，这是不应用温度系数的原始 logits 的 log_softmax，供后续 GRPO 更新使用
        token_logps = F.log_softmax(next_logits, dim=-1)  # (B*G, vocab_size)
        chosen_logps = token_logps.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)  # 每个生成所选中的 token 的 logp (B*G,)

        was_finished = finished.clone()
        next_tokens = next_tokens.masked_fill(was_finished, pad_token_id)  # 将已生成完毕的样本的 token 填充为 pad_token_id
        chosen_logps = chosen_logps.masked_fill(was_finished, 0.0)  # 将已生成完毕的样本的 logp 填充为 0
        current_token_mask = (~was_finished).long()  # 当前 token 对应的 attention mask (B*G,)

        generated_tokens.append(next_tokens)
        generated_logps.append(chosen_logps)
        generated_token_masks.append(current_token_mask)

        cur_token = next_tokens.unsqueeze(-1)  # (B*G, 1)
        finished = finished | (next_tokens == eos_token_id)  # 更新生成完毕标记

        current_attention_mask = torch.cat([current_attention_mask, current_token_mask.unsqueeze(-1)], dim=1)  # 更新 attention mask
        current_position_ids = current_attention_mask.sum(dim=1, keepdim=True) - 1  # 根据 mask 更新 position_ids

        if finished.all():
            break

    del past_key_values

    gen_ids = torch.stack(generated_tokens, dim=1)
    gen_logps = torch.stack(generated_logps, dim=1)
    gen_token_mask = torch.stack(generated_token_masks, dim=1)
    old_logps = gen_logps.sum(dim=-1) / gen_token_mask.sum(dim=-1).clamp_min(1).to(gen_logps.dtype)  # 对 logp 进行长度归一化

    # ------- step 2. 构建完整序列 ---------
    flat_response_texts = []
    flat_full_ids = []
    flat_attention_masks = []
    flat_labels = []

    for flat_idx in range(expanded_batch_size):
        source_idx = flat_idx // group_size
        prompt_len = prompt_lengths[source_idx]
        gen_valid_len = int(gen_token_mask[flat_idx].sum().item())

        prompt_row = prompt_ids[source_idx, -prompt_len:]
        gen_row = gen_ids[flat_idx, :gen_valid_len]
        full_row = torch.cat([prompt_row, gen_row], dim=0)  # 完整序列
        attn_row = torch.ones(full_row.size(0), dtype=torch.long, device=device)
        label_row = full_row.clone()
        label_row[:prompt_len] = -100

        response_ids = gen_row.tolist()
        if eos_token_id in response_ids:
            response_ids = response_ids[:response_ids.index(eos_token_id)]
        flat_response_texts.append(tokenizer.decode(response_ids, skip_special_tokens=False))
        flat_full_ids.append(full_row)
        flat_attention_masks.append(attn_row)
        flat_labels.append(label_row)

    # 填充成统一长度
    max_total_len = max(row.size(0) for row in flat_full_ids)
    padded_full_ids = []
    padded_attn_masks = []
    padded_labels = []

    for ids, mask, lbl in zip(flat_full_ids, flat_attention_masks, flat_labels):
        pad_len = max_total_len - ids.size(0)
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=pad_token_id)
            mask = F.pad(mask, (0, pad_len), value=0)
            lbl = F.pad(lbl, (0, pad_len), value=-100)
        padded_full_ids.append(ids.unsqueeze(0))
        padded_attn_masks.append(mask.unsqueeze(0))
        padded_labels.append(lbl.unsqueeze(0))

    all_response_texts = [
        flat_response_texts[i * group_size:(i + 1) * group_size]
        for i in range(batch_size)
    ]

    return {
        "full_ids": torch.cat(padded_full_ids, dim=0),  # (B*G, max_total_len)
        "attention_mask": torch.cat(padded_attn_masks, dim=0),  # (B*G, max_total_len)
        "labels": torch.cat(padded_labels, dim=0),  # (B*G, max_total_len)
        "old_logps": old_logps,  # (B*G,)
        "response_texts": all_response_texts,  # list[list[str]], shape (B, G)
    }


# -------------------------------------------【阶段训练】------------------------------------------- #
# Mid-training: 在任务领域上继续预训练
def _run_mid_training(
    model,
    args,
    device: torch.device,
    enable_amp: bool,
    autocast_dtype: torch.dtype | None,
    scaler: GradScaler | None,
    save_path: str,
) -> None:
    print("=" * 60)
    print("Starting mid-training (continued pretraining on domain data) ...")
    print("=" * 60)

    dataset = PreTrainDataset(
        file_path=args.mid_training_data_path,
        max_seq_len=args.max_seq_len,
    )
    mid_batch_size = args.mid_batch_size if args.mid_batch_size is not None else args.max_batch_size
    dataloader = DataLoader(dataset, batch_size=mid_batch_size, shuffle=True)

    iter_per_epoch = len(dataloader)
    total_iters = args.mid_epochs * iter_per_epoch
    warmup_iters = int(total_iters * args.mid_warmup_ratio)
    lr_decay_iters = int(total_iters * args.mid_lr_decay_ratio)
    if lr_decay_iters <= warmup_iters:
        raise ValueError("Mid-training: lr_decay_iters must be greater than warmup_iters.")

    optimizer = configure_optimizer(
        model=model,
        weight_decay=args.mid_weight_decay,
        learning_rate=args.mid_max_lr,
        betas=args.betas,
        device_type="cuda",
    )

    model.train()
    start_time = time.time()

    for epoch in range(args.mid_epochs):
        for step, input_ids in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            it = epoch * iter_per_epoch + step + 1
            lr = get_lr(it, max_lr=args.mid_max_lr, min_lr=args.mid_min_lr,
                        warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = input_ids.to(device)
            with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                rest_time = spend_time / it * total_iters - spend_time
                print(
                    f"[Mid-Training] Epoch: {epoch + 1}/{args.mid_epochs} | "
                    f"Step: {step + 1}/{iter_per_epoch} | "
                    f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                    f"Remaining: {datetime.timedelta(seconds=max(rest_time, 0))}"
                )

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Mid-training finished. Model saved to: {save_path}")
    print("=" * 60)


# 冷启动 SFT: 在 GRPO 训练前先用标注数据做 SFT，让模型学会输出格式
def _run_cold_start_sft(
    model,
    dataset,
    args,
    device: torch.device,
    enable_amp: bool,
    autocast_dtype: torch.dtype | None,
    scaler: GradScaler | None,
    save_path: str,
) -> None:
    print("=" * 60)
    print("Starting cold-start SFT ...")
    print("=" * 60)

    sft_batch_size = args.sft_batch_size if args.sft_batch_size is not None else args.max_batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=sft_batch_size,
        shuffle=True,
        collate_fn=dataset.sft_collate_fn,
    )

    iter_per_epoch = len(dataloader)
    total_iters = args.sft_epochs * iter_per_epoch
    warmup_iters = int(total_iters * args.sft_warmup_ratio)
    lr_decay_iters = int(total_iters * args.sft_lr_decay_ratio)
    if lr_decay_iters <= warmup_iters:
        raise ValueError("Cold-start SFT: lr_decay_iters must be greater than warmup_iters.")

    optimizer = configure_optimizer(
        model=model,
        weight_decay=args.sft_weight_decay,
        learning_rate=args.sft_max_lr,
        betas=args.betas,
        device_type="cuda",
    )

    model.train()
    start_time = time.time()

    for epoch in range(args.sft_epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            it = epoch * iter_per_epoch + step + 1
            lr = get_lr(it, max_lr=args.sft_max_lr, min_lr=args.sft_min_lr,
                        warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)

            with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=labels,
                )
                loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.sft_max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.sft_max_grad_norm)
                optimizer.step()

            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                rest_time = spend_time / it * total_iters - spend_time
                print(
                    f"[Cold-Start SFT] Epoch: {epoch + 1}/{args.sft_epochs} | "
                    f"Step: {step + 1}/{iter_per_epoch} | "
                    f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                    f"Remaining: {datetime.timedelta(seconds=max(rest_time, 0))}"
                )

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Cold-start SFT finished. Model saved to: {save_path}")
    print("=" * 60)


# -------------------------------------------【训练函数】------------------------------------------- #
def train_process(args):

    # ------------------ 1. 基础设置 ------------------
    device = torch.device("cuda")

    _resolve_model_paths(args)
    args.max_seq_len = _resolve_max_seq_len(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # ------------------ 2. 混合精度配置 ------------------
    scaler = None
    autocast_dtype = None
    enable_amp = False
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_dtype = torch.float16
        enable_amp = True
        print("Using FP16 mixed precision training")
    elif args.precision == "bf16":
        autocast_dtype = torch.bfloat16
        enable_amp = True
        print("Using BF16 mixed precision training")
    else:
        print("Using FP32 precision training")

    # ------------------ 3. 配置输出目录 ------------------
    os.makedirs(args.output_path, exist_ok=True)
    model_name = f"grpo_{args.model_name}"
    current_train_path = create_folder(os.path.join(args.output_path, model_name))
    log_dir = os.path.join(current_train_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    save_args(args, os.path.join(current_train_path, f"{model_name}_training_args.json"))
    print(f"Training arguments saved to: {os.path.join(current_train_path, f'{model_name}_training_args.json')}")
    print(f"Support models: {support_models}")

    # ------------------ 4. Mid-training（可选）------------------
    if args.mid_training:
        print(f"Loading model for mid-training: {args.base_model_path}")
        model = _load_model(args.model_name, args.base_model_path, device, trainable=True)
        print(f"Model info: {json.dumps(get_model_info(model)[1], indent=2)}")

        mid_save_path = os.path.join(current_train_path, "mid_training")
        _run_mid_training(model, args, device, enable_amp, autocast_dtype, scaler, mid_save_path)

        del model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # 后续阶段使用 mid-training 后的模型
        args.base_model_path = mid_save_path

    # ------------------ 5. 冷启动 SFT（可选）------------------
    if args.cold_start_sft:
        grpo_dataset = GRPODataset(
            file_path=args.sft_data_path,  # 冷启动数据集
            max_seq_len=args.max_seq_len,
            tokenizer=tokenizer,
        )
        if len(grpo_dataset) == 0:
            raise ValueError("No valid GRPO samples found for cold-start SFT.")

        print(f"Loading model for cold-start SFT: {args.base_model_path}")
        model = _load_model(args.model_name, args.base_model_path, device, trainable=True)
        print(f"Model info: {json.dumps(get_model_info(model)[1], indent=2)}")

        sft_save_path = os.path.join(current_train_path, "cold_start_sft")
        _run_cold_start_sft(model, grpo_dataset, args, device, enable_amp, autocast_dtype, scaler, sft_save_path)

        del model, grpo_dataset
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # 后续阶段使用冷启动 SFT 后的模型
        args.base_model_path = sft_save_path

    # ------------------ 6. GRPO 训练 ------------------
    if args.no_grpo:
        print("--no_grpo is set, skipping GRPO training.")
        print(f"Final model path: {args.base_model_path}")
        writer.close()
        return

    print("=" * 60)
    print("Starting GRPO training ...")
    print("=" * 60)

    # 准备数据
    grpo_dataset = GRPODataset(
        file_path=args.grpo_data_path,  # GRPO 数据集
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
    )
    if len(grpo_dataset) == 0:
        raise ValueError("No valid GRPO samples found for GRPO training.")

    prompt_dataloader = DataLoader(
        grpo_dataset,
        batch_size=args.max_batch_size,
        shuffle=True,
        collate_fn=grpo_dataset.prompt_collate_fn,
    )

    # 加载 policy model（可训练）和 reference model（冻结）
    print(f"Loading policy model: {args.base_model_path}")
    policy_model = _load_model(args.model_name, args.base_model_path, device, trainable=True)
    print(f"Policy model info: {json.dumps(get_model_info(policy_model)[1], indent=2)}")

    print(f"Loading reference model: {args.base_model_path}")
    ref_model = _load_model(args.model_name, args.base_model_path, device, trainable=False)

    # 优化器和学习率调度
    iter_per_epoch = len(prompt_dataloader)
    total_iters = args.grpo_epochs * iter_per_epoch
    warmup_iters = int(total_iters * args.grpo_warmup_ratio)
    lr_decay_iters = int(total_iters * args.grpo_lr_decay_ratio)
    if lr_decay_iters <= warmup_iters:
        raise ValueError("GRPO: lr_decay_iters must be greater than warmup_iters.")
    if lr_decay_iters > total_iters:
        raise ValueError("GRPO: lr_decay_iters must be less than or equal to total_iters.")

    optimizer = configure_optimizer(
        model=policy_model,
        weight_decay=args.grpo_weight_decay,
        learning_rate=args.grpo_max_lr,
        betas=args.betas,
        device_type="cuda",
    )

    total_loss = []
    total_surrogate_loss = []
    total_kl = []
    total_mean_reward = []
    total_format_reward = []
    total_think_reward = []
    total_parse_reward = []
    total_correct_reward = []
    start_time = time.time()

    # GRPO 训练循环
    for epoch in range(args.grpo_epochs):
        for step, batch in enumerate(prompt_dataloader):
            optimizer.zero_grad(set_to_none=True)

            it = epoch * iter_per_epoch + step + 1
            lr = get_lr(it, max_lr=args.grpo_max_lr, min_lr=args.grpo_min_lr,
                        warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            prompt_ids = batch["input_ids"].to(device)
            prompt_mask = batch["attention_mask"].to(device)
            prompt_lengths = batch["prompt_lengths"]
            batch_size = len(prompt_lengths)
            remaining_context = min(args.max_seq_len - prompt_len for prompt_len in prompt_lengths)
            effective_max_new_tokens = min(args.max_new_tokens, max(remaining_context, 0))  # 剩余可生成的最大长度

            if effective_max_new_tokens <= 0:
                if step % args.log_interval == 0:
                    print(
                        f"Epoch: {epoch + 1}/{args.grpo_epochs} | Step: {step + 1}/{iter_per_epoch} | "
                        "Skipped rollout because prompts already filled the context window."
                    )
                continue

            # === Phase 1: 生成补全（无梯度）===
            completions = _generate_completions(
                model=policy_model,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                prompt_lengths=prompt_lengths,
                group_size=args.group_size,
                max_new_tokens=effective_max_new_tokens,
                temperature=args.temperature,
                tokenizer=tokenizer,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                enable_amp=enable_amp,
                autocast_dtype=autocast_dtype,
            )

            # === Phase 2: 计算奖励 ===
            ground_truth_responses = batch["ground_truth_responses"]
            rewards = torch.zeros(batch_size, args.group_size, device=device)
            batch_reward_details = {"format": 0.0, "think_length": 0.0, "parseable": 0.0, "correctness": 0.0}  # 用于输出到日志
            for i in range(batch_size):
                for g in range(args.group_size):
                    r, details = _compute_reward(
                        completions["response_texts"][i][g],
                        ground_truth_responses[i],
                    )
                    rewards[i, g] = r
                    for k in batch_reward_details:
                        batch_reward_details[k] += details[k]
            # 取平均
            n_completions = batch_size * args.group_size
            for k in batch_reward_details:
                batch_reward_details[k] /= n_completions

            # 组内归一化得到 advantages
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
            advantages = (rewards - mean_r) / std_r  # (B, G)
            advantages = advantages.view(-1).detach()  # (B*G,)

            # === Phase 3: 计算 reference logps（无梯度）===
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):
                    ref_logps = _compute_response_logps(
                        ref_model,
                        completions["full_ids"],
                        completions["attention_mask"],
                        completions["labels"],
                        device,
                    )  # (B*G,)

            # === Phase 4: 计算 policy logps（有梯度）并计算 GRPO loss ===
            policy_model.train()
            with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):
                new_logps = _compute_response_logps(
                    policy_model,
                    completions["full_ids"],
                    completions["attention_mask"],
                    completions["labels"],
                    device,
                )  # (B*G,)

                old_logps = completions["old_logps"].to(device).detach()  # (B*G,)

                # 计算 ratio
                # NOTE: 当前每个 batch 一次 rollout，一次更新，因此理论上 ratio 为 1
                # 通常也可以每个 batch 进行一个 rollout 后，policy model 多次前向和更新，此时 ratio 会偏离 1，重要性采样开始起作用
                ratio = torch.exp(new_logps - old_logps)
                clipped_ratio = torch.clamp(ratio, 1.0 - args.grpo_clip_eps, 1.0 + args.grpo_clip_eps)
                surrogate = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                # KL penalty
                log_ratio = ref_logps.detach() - new_logps
                kl = (torch.exp(log_ratio) - log_ratio - 1).mean()  # GRPO 所采用的 KL 估计器，如果每个 batch 只更新一次，它等价于 kl = (new_logps - ref_logps.detach()).mean()

                loss = surrogate + args.kl_coeff * kl

            # === Phase 5: 反向传播 ===
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                optimizer.step()

            # === 记录指标 ===
            loss_val = float(loss.item())
            surrogate_val = float(surrogate.item())
            kl_val = float(kl.item())
            mean_reward = float(rewards.mean().item())

            total_loss.append(loss_val)
            total_surrogate_loss.append(surrogate_val)
            total_kl.append(kl_val)
            total_mean_reward.append(mean_reward)
            total_format_reward.append(batch_reward_details["format"])
            total_think_reward.append(batch_reward_details["think_length"])
            total_parse_reward.append(batch_reward_details["parseable"])
            total_correct_reward.append(batch_reward_details["correctness"])

            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                rest_time = spend_time / it * total_iters - spend_time

                print(
                    f"Epoch: {epoch + 1}/{args.grpo_epochs} | Step: {step + 1}/{iter_per_epoch} | "
                    f"Loss: {loss_val:.4f} | Surrogate: {surrogate_val:.4f} | KL: {kl_val:.4f} | "
                    f"Mean Reward: {mean_reward:.4f} | "
                    f"Format Reward: {batch_reward_details['format']:.3f} | "
                    f"Think Reward: {batch_reward_details['think_length']:.3f} | "
                    f"Parse Reward: {batch_reward_details['parseable']:.3f} | "
                    f"Correct Reward: {batch_reward_details['correctness']:.3f} | "
                    f"LR: {lr:.2e} | s/it: {spend_time / it:.4f} | "
                    f"Remaining: {datetime.timedelta(seconds=max(rest_time, 0))}"
                )

                writer.add_scalar("GRPO/Loss", loss_val, it)
                writer.add_scalar("GRPO/Surrogate Loss", surrogate_val, it)
                writer.add_scalar("GRPO/KL Divergence", kl_val, it)
                writer.add_scalar("GRPO/Mean Reward", mean_reward, it)
                writer.add_scalar("GRPO/Reward Format", batch_reward_details["format"], it)
                writer.add_scalar("GRPO/Reward Think Length", batch_reward_details["think_length"], it)
                writer.add_scalar("GRPO/Reward Parseable", batch_reward_details["parseable"], it)
                writer.add_scalar("GRPO/Reward Correctness", batch_reward_details["correctness"], it)
                writer.add_scalar("Learning Rate", lr, it)

    # ------------------ 7. 保存模型 ------------------
    plot_curve_grpo(
        total_loss=total_loss,
        total_surrogate_loss=total_surrogate_loss,
        total_kl=total_kl,
        total_mean_reward=total_mean_reward,
        total_format_reward=total_format_reward,
        total_think_reward=total_think_reward,
        total_parse_reward=total_parse_reward,
        total_correct_reward=total_correct_reward,
        save_path=os.path.join(current_train_path, f"{model_name}_curve.png"),
    )
    print(f"Curve saved to: {os.path.join(current_train_path, f'{model_name}_curve.png')}")

    policy_model.save_pretrained(current_train_path)
    print(f"Model saved to: {current_train_path}")
    writer.close()

    # 释放 reference model
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------------------------【主函数】------------------------------------------- #
def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. GPU is required for training.")
        return

    train_process(args)


if __name__ == "__main__":
    main()
