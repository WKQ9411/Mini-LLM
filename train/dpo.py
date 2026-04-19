import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from data_loader import DPODataset
from mini_models import get_model_and_config, get_model_info, list_models
from utils import (
    configure_optimizer,
    create_folder,
    get_lr,
    load_args,
    plot_curve_dpo,
    save_args,
)


root_path = Path(__file__).parent.parent
support_models = ", ".join(list_models())


# -------------------------------------------【参数解析】------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="DPO Mini-LLM")

    # 模型与训练精度
    parser.add_argument("--model_name", type=str, required=True, help=f"Mini model names, support: {support_models}")
    parser.add_argument("--precision", type=str, default="bf16", help="Mixed precision training: default bf16, options are fp32 or fp16")
    parser.add_argument("--base_model_path", type=str, default=None, help="Policy model checkpoint path, default: output/sft_{model_name}")
    parser.add_argument("--reference_model_path", type=str, default=None, help="Reference model checkpoint path, default follows base_model_path")

    # ModelArgs 设置
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum sequence length for training samples. If not set, will be inferred from model config or training args.")
    parser.add_argument("--max_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_lr", type=float, default=1e-6, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=None, help="Number of warmup iterations")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup iteration ratio")
    parser.add_argument("--lr_decay_iters", type=int, default=None, help="Number of learning rate decay iterations")
    parser.add_argument("--lr_decay_ratio", type=float, default=0.98, help="Learning rate decay iteration ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="Beta parameters for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    
    # DPO β 设置，在原始目标函数，即 max (Reward + beta * KL) 中，较大的 β 意味着更强的 KL 正则化，policy 会被约束不偏离 reference 过多，即更保守
    # 但经过 DPO 推到后，DPO loss = -log(σ(β * (chosen_logp - rejected_logp)))，β 的作用改变了，较大的 β 意味着更激进的更新 
    # 由于我们的模型参数量很小，为了避免 DPO 训崩，使用较小的 β 和学习率，此外，可选 DPOP、冻结参数、预先对 chosen 进行 SFT 等优化措施
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta coefficient")
    
    # DPO 数据集的得分范围过滤，得分范围为 1-5 分，分数越高，代表 chosen 和 rejected 之间的可区分度越高
    parser.add_argument("--min_score", type=int, default=3, help="Minimum score of samples used for training")
    parser.add_argument("--max_score", type=int, default=5, help="Maximum score of samples used for training")
    
    # [可选] 冻结层数，如果不为 0，则冻结 embedding, lm_head, 和前 n 层 transformer layers
    parser.add_argument("--freeze_layers", type=int, default=0, help="Freeze embedding, lm_head, and the first n transformer layers of the policy model")
    
    # [可选] DPOP 设置，在 DPO 基础上增加对 policy chosen logp 的约束
    parser.add_argument("--loss_type", type=str, default="dpo", choices=["dpo", "dpop"], help="Preference optimization loss type: standard DPO or DPOP (DPO-Positive)")
    parser.add_argument("--dpop_lambda", type=float, default=1.0, help="Positive hinge penalty weight used only when --loss_type dpop")

    # [可选] SFT Warmup 设置（在 DPO 前先对 chosen 数据做 SFT，减小分布偏差）
    parser.add_argument("--sft_warmup", action="store_true", help="Enable SFT warmup on chosen data before DPO training")
    parser.add_argument("--sft_epochs", type=int, default=1, help="SFT warmup training epochs")
    parser.add_argument("--sft_max_lr", type=float, default=3e-5, help="SFT warmup max learning rate")
    parser.add_argument("--sft_min_lr", type=float, default=1e-6, help="SFT warmup min learning rate")
    parser.add_argument("--sft_warmup_ratio", type=float, default=0.05, help="SFT warmup LR warmup ratio")
    parser.add_argument("--sft_lr_decay_ratio", type=float, default=0.98, help="SFT warmup LR decay ratio")
    parser.add_argument("--sft_weight_decay", type=float, default=0.1, help="SFT warmup weight decay")
    parser.add_argument("--sft_max_grad_norm", type=float, default=1.0, help="SFT warmup max gradient norm")
    parser.add_argument("--sft_batch_size", type=int, default=None, help="SFT warmup batch size (default: same as max_batch_size)")

    # 路径与日志设置
    parser.add_argument("--dpo_data_path", type=str, default=f"{root_path}/data/dpo_data/dpo.jsonl", help="Path to dpo jsonl dataset")
    parser.add_argument("--tokenizer_path", type=str, default=f"{root_path}/mini_tokenizer", help="Tokenizer path used by DPO dataset")
    parser.add_argument("--output_path", type=str, default=f"{root_path}/output", help="Model output directory")
    parser.add_argument("--log_interval", type=int, default=50, help="Training log print interval")

    args = parser.parse_args()
    if args.min_score > args.max_score:
        raise ValueError("min_score must be less than or equal to max_score.")
    if args.freeze_layers < 0:
        raise ValueError("freeze_layers must be greater than or equal to 0.")
    if args.dpop_lambda < 0:
        raise ValueError("dpop_lambda must be greater than or equal to 0.")
    return args


# -------------------------------------------【辅助函数】------------------------------------------- #
# 解析模型路径
def _resolve_model_paths(args):
    if args.base_model_path is None:
        args.base_model_path = str(root_path / f"output/sft_{args.model_name}")
    if args.reference_model_path is None:
        args.reference_model_path = args.base_model_path

    if not os.path.exists(args.base_model_path):
        raise FileNotFoundError(f"Policy model path not found: {args.base_model_path}")
    if not os.path.exists(args.reference_model_path):
        raise FileNotFoundError(f"Reference model path not found: {args.reference_model_path}")


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
def _adjust_model_for_dpo(model, model_name: str) -> None:
    if model_name == "mini_deepseekv3":
        model.config.use_mtp = False
        model.config.use_noaux_load_balance = False
        model.config.use_seq_aux = False


# 冻结部分 policy 模型层
def _freeze_policy_layers(model, freeze_layers: int) -> None:
    if freeze_layers <= 0:
        return

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Policy model does not expose `model.layers`, cannot freeze the requested transformer layers.")

    total_layers = len(model.model.layers)
    if freeze_layers > total_layers:
        raise ValueError(
            f"Requested freeze_layers={freeze_layers}, but model only has {total_layers} layers. "
            "Freezing all transformer layers would leave no trainable backbone for DPO."
        )
    actual_freeze_layers = int(freeze_layers)

    if hasattr(model.model, "embed_tokens") and model.model.embed_tokens is not None:
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

    if hasattr(model, "lm_head") and model.lm_head is not None:
        for param in model.lm_head.parameters():
            param.requires_grad = False

    for layer in model.model.layers[:actual_freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    print(
        f"Froze policy model embedding, lm_head, and the first "
        f"{actual_freeze_layers}/{total_layers} transformer layers."
    )


# 加载模型并设置训练状态
def _load_model(
    model_name: str,
    model_path: str,
    device: torch.device,
    trainable: bool,
    freeze_layers: int = 0,
):
    Model, _ = get_model_and_config(model_name)
    model = Model.from_pretrained(model_path).to(device)
    _adjust_model_for_dpo(model, model_name)

    if trainable:
        model.train()
        _freeze_policy_layers(model, freeze_layers)
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


# 计算序列的总 logprob 和有效 token 数量
def _compute_sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float()  # (batch_size, seq_len-1, vocab_size)
    shift_labels = labels[:, 1:]  # (batch_size, seq_len-1)
    loss_mask = shift_labels.ne(-100)  # ne 相当于 not equal，生成一个布尔掩码，标记哪些位置的标签不是 -100

    # 计算真实标签对应的 logp
    # gather 表示在词表维度上，按照 safe_labels 中的 token id 索引，取出正确的 logp
    safe_labels = shift_labels.masked_fill(~loss_mask, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)  # (batch_size, seq_len-1)
    token_logps = token_logps * loss_mask  # 将无效位置的 logp 置零

    seq_logps = token_logps.sum(dim=-1)  # (batch_size,) 每个序列的总 logp，相当于 log π(y|x)
    token_counts = loss_mask.sum(dim=-1)  # (batch_size,) 每个序列的有效 token 数量
    return seq_logps, token_counts


# 前向辅助函数，用于产生 logps
def _forward_pair_logps(model, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:

    # 读取 chosen 数据
    chosen_input_ids = batch["chosen_input_ids"].to(device)
    chosen_attention_mask = batch["chosen_attention_mask"].to(device)
    chosen_position_ids = batch["chosen_position_ids"].to(device)
    chosen_labels = batch["chosen_labels"].to(device)

    # 读取 rejected 数据
    rejected_input_ids = batch["rejected_input_ids"].to(device)
    rejected_attention_mask = batch["rejected_attention_mask"].to(device)
    rejected_position_ids = batch["rejected_position_ids"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)

    # 在 batch 维度上拼接 chosen 和 rejected 数据，以便一次前向计算
    concat_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
    concat_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
    concat_position_ids = torch.cat([chosen_position_ids, rejected_position_ids], dim=0)
    concat_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

    outputs: CausalLMOutputWithPast = model(
        input_ids=concat_input_ids,
        attention_mask=concat_attention_mask,
        position_ids=concat_position_ids,
    )
    seq_logps, token_counts = _compute_sequence_logps(outputs.logits, concat_labels)
    # 对 logp 进行长度归一化
    seq_logps = seq_logps / token_counts.clamp_min(1).to(seq_logps.dtype)

    half = chosen_input_ids.size(0)
    return (
        seq_logps[:half],  # chosen
        seq_logps[half:],  # rejected
    )


# 预计算 reference logp 以减小显存压力
def _build_ref_logp_cache(
    reference_model,
    dataloader: DataLoader,
    device: torch.device,
    enable_amp: bool,
    autocast_dtype: torch.dtype | None,
) -> dict[str, tuple[float, float]]:
    """
    先使用 reference model 对全部样本做一次前向, 缓存每个 sample_id 对应的 chosen/rejected logp
    缓存放在 CPU 内存, 避免占用训练阶段显存
    """
    ref_logp_cache: dict[str, tuple[float, float]] = {}

    reference_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Precomputing reference logps"):
            with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):
                ref_chosen_logps, ref_rejected_logps = _forward_pair_logps(reference_model, batch, device)

            sample_ids = batch["sample_ids"]
            for sample_id, chosen_logp, rejected_logp in zip(sample_ids, ref_chosen_logps, ref_rejected_logps):
                key = str(sample_id)
                if key in ref_logp_cache:
                    raise ValueError(f"Duplicated sample_id found in dataset: {sample_id}")
                ref_logp_cache[key] = (
                    float(chosen_logp.detach().cpu().item()),
                    float(rejected_logp.detach().cpu().item()),
                )

    return ref_logp_cache


# 从缓存中获取 reference logp
def _get_ref_logps_from_cache(
    sample_ids: list[Any],
    ref_logp_cache: dict[str, tuple[float, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_chosen_vals = []
    ref_rejected_vals = []

    for sample_id in sample_ids:
        key = str(sample_id)
        if key not in ref_logp_cache:
            raise KeyError(f"sample_id not found in reference logp cache: {sample_id}")
        chosen_logp, rejected_logp = ref_logp_cache[key]
        ref_chosen_vals.append(chosen_logp)
        ref_rejected_vals.append(rejected_logp)

    ref_chosen_logps = torch.tensor(ref_chosen_vals, dtype=dtype, device=device)
    ref_rejected_logps = torch.tensor(ref_rejected_vals, dtype=dtype, device=device)
    return ref_chosen_logps, ref_rejected_logps


# 计算偏好损失，包括 DPO 和 DPOP
def _compute_preference_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    loss_type: str = "dpo",
    dpop_lambda: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    reward_margin = chosen_reward - rejected_reward

    dpo_loss = -F.logsigmoid(reward_margin)
    dpop_penalty = torch.zeros_like(dpo_loss)

    if loss_type == "dpop":
        dpop_penalty = dpop_lambda * torch.relu(ref_chosen_logps - policy_chosen_logps)  # 如果 policy 的 chosen logp 相较 reference 变小，则加一个惩罚项
        loss = (dpo_loss + dpop_penalty).mean()
    elif loss_type == "dpo":
        loss = dpo_loss.mean()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss, chosen_reward, rejected_reward, reward_margin, dpop_penalty


# 预先对 chosen 数据进行 SFT 以减小分布差异
def _run_sft_warmup(
    model,
    dataset,
    args,
    device: torch.device,
    enable_amp: bool,
    autocast_dtype: torch.dtype | None,
    scaler: GradScaler | None,
    save_path: str,
) -> None:
    """
    在 DPO 训练前, 使用 chosen 数据对 policy model 做 SFT warmup
    训练结束后将模型权重保存到 save_path, 用于后续作为 reference model
    """
    print("=" * 60)
    print("Starting SFT warmup on chosen data ...")
    print("=" * 60)

    sft_batch_size = args.sft_batch_size if args.sft_batch_size is not None else args.max_batch_size
    sft_dataloader = DataLoader(
        dataset,
        batch_size=sft_batch_size,
        shuffle=True,
        collate_fn=dataset.sft_collate_fn,
    )

    iter_per_epoch = len(sft_dataloader)
    total_iters = args.sft_epochs * iter_per_epoch
    warmup_iters = int(total_iters * args.sft_warmup_ratio)
    lr_decay_iters = int(total_iters * args.sft_lr_decay_ratio)
    if lr_decay_iters <= warmup_iters:
        raise ValueError("SFT: lr_decay_iters must be greater than warmup_iters.")

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
        for step, batch in enumerate(sft_dataloader):
            optimizer.zero_grad(set_to_none=True)

            it = epoch * iter_per_epoch + step + 1
            lr = get_lr(
                it,
                max_lr=args.sft_max_lr,
                min_lr=args.sft_min_lr,
                warmup_iters=warmup_iters,
                lr_decay_iters=lr_decay_iters,
            )
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
                    f"[SFT Warmup] Epoch: {epoch + 1}/{args.sft_epochs} | "
                    f"Step: {step + 1}/{iter_per_epoch} | "
                    f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                    f"Remaining: {datetime.timedelta(seconds=max(rest_time, 0))}"
                )

    # 保存 SFT warmup 后的模型
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"SFT warmup finished. Model saved to: {save_path}")
    print("=" * 60)


# -------------------------------------------【训练函数】------------------------------------------- #
def train_process(args):

    # ------------------ 1. 设置 ------------------
    device = torch.device("cuda")

    _resolve_model_paths(args)
    args.max_seq_len = _resolve_max_seq_len(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # ------------------ 2. 数据准备 ------------------
    dataset = DPODataset(
        file_path=args.dpo_data_path,
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
        min_score=args.min_score,
        max_score=args.max_score,
    )
    if len(dataset) == 0:
        raise ValueError("No valid DPO samples found after filtering.")

    # 预计算 reference logp 使用固定顺序遍历；训练阶段仍使用 shuffle
    ref_dataloader = DataLoader(
        dataset,
        batch_size=args.max_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.max_batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # ------------------ 3. 混合精度配置 ------------------
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

    # ------------------ 4. 配置输出目录 ------------------
    os.makedirs(args.output_path, exist_ok=True)
    model_name = f"{args.loss_type}_{args.model_name}"
    current_train_path = create_folder(os.path.join(args.output_path, model_name))
    log_dir = os.path.join(current_train_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 保存本次训练配置
    save_args(args, os.path.join(current_train_path, f"{model_name}_training_args.json"))
    print(f"Training arguments saved to: {os.path.join(current_train_path, f'{model_name}_training_args.json')}")

    print(f"Support models: {support_models}")

    # ------------------ 5. SFT Warmup（可选）------------------
    if args.sft_warmup:
        # 先加载 policy model 做 SFT warmup
        print(f"Loading policy model for SFT warmup: {args.base_model_path}")
        policy_model = _load_model(
            args.model_name,
            args.base_model_path,
            device,
            trainable=True,
            freeze_layers=args.freeze_layers,
        )
        print(f"Policy model info: {json.dumps(get_model_info(policy_model)[1], indent=2)}")

        sft_save_path = os.path.join(current_train_path, "sft_warmup")
        _run_sft_warmup(
            model=policy_model,
            dataset=dataset,
            args=args,
            device=device,
            enable_amp=enable_amp,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            save_path=sft_save_path,
        )

        # SFT warmup 后的模型作为 reference model 计算 ref logps
        policy_model.eval()
        for param in policy_model.parameters():
            param.requires_grad = False

        print("Using SFT-warmed model as reference model for DPO ...")
        ref_logp_cache = _build_ref_logp_cache(
            reference_model=policy_model,
            dataloader=ref_dataloader,
            device=device,
            enable_amp=enable_amp,
            autocast_dtype=autocast_dtype,
        )

        # 释放 reference 副本，重新从 SFT checkpoint 加载可训练的 policy model
        del policy_model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        print(f"Loading SFT-warmed policy model for DPO: {sft_save_path}")
        policy_model = _load_model(
            args.model_name,
            sft_save_path,
            device,
            trainable=True,
            freeze_layers=args.freeze_layers,
        )

    else:
        # 先加载 reference model 缓存 logps，再加载 policy model
        print(f"Loading reference model: {args.reference_model_path}")
        reference_model = _load_model(args.model_name, args.reference_model_path, device, trainable=False)
        print(f"Reference model info: {json.dumps(get_model_info(reference_model)[1], indent=2)}")

        ref_logp_cache = _build_ref_logp_cache(
            reference_model=reference_model,
            dataloader=ref_dataloader,
            device=device,
            enable_amp=enable_amp,
            autocast_dtype=autocast_dtype,
        )

        del reference_model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        print(f"Loading policy model: {args.base_model_path}")
        policy_model = _load_model(
            args.model_name,
            args.base_model_path,
            device,
            trainable=True,
            freeze_layers=args.freeze_layers,
        )

    if len(ref_logp_cache) != len(dataset):
        raise ValueError(
            f"Reference logp cache size mismatch. cache={len(ref_logp_cache)}, dataset={len(dataset)}"
        )
    print(f"Reference logp precompute finished. Cached samples: {len(ref_logp_cache)}")
    print(f"Using preference loss: {args.loss_type.upper()}" + (f" (lambda={args.dpop_lambda})" if args.loss_type == "dpop" else ""))
    print(f"Policy model info: {json.dumps(get_model_info(policy_model)[1], indent=2)}")

    # ------------------ 6. DPO 训练配置 ------------------
    iter_per_epoch = len(train_dataloader)
    total_iters = args.epochs * iter_per_epoch
    warmup_iters = int(total_iters * args.warmup_ratio) if args.warmup_iters is None else args.warmup_iters  # 预热迭代次数
    lr_decay_iters = int(total_iters * args.lr_decay_ratio) if args.lr_decay_iters is None else args.lr_decay_iters  # 衰减迭代次数
    if lr_decay_iters <= warmup_iters:
        raise ValueError("lr_decay_iters must be greater than warmup_iters.")
    if lr_decay_iters > total_iters:
        raise ValueError("lr_decay_iters must be less than or equal to total_iters.")

    optimizer = configure_optimizer(
        model=policy_model,
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        betas=args.betas,
        device_type="cuda",
    )

    total_loss = []
    total_reward_margin = []
    total_chosen_reward = []
    total_rejected_reward = []
    total_chosen_logp = []
    total_rejected_logp = []
    total_dpop_penalty = []
    start_time = time.time()

    # ------------------ 7. DPO 训练循环 ------------------
    for epoch in range(args.epochs):
        policy_model.train()

        for step, batch in enumerate(train_dataloader):

            # 清零梯度
            optimizer.zero_grad(set_to_none=True)

            it = epoch * iter_per_epoch + step + 1  # 当前全局迭代次数
            lr = get_lr(
                it,
                max_lr=args.max_lr,
                min_lr=args.min_lr,
                warmup_iters=warmup_iters,
                lr_decay_iters=lr_decay_iters,
            )  # 获取当前学习率
            for param_group in optimizer.param_groups:  # 将新的学习率值应用到优化器中
                param_group["lr"] = lr

            with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):  # 使用混合精度训练
                # 前向计算 policy logp
                policy_chosen_logps, policy_rejected_logps = _forward_pair_logps(policy_model, batch, device)

                # 从缓存中读取 reference logp
                ref_chosen_logps, ref_rejected_logps = _get_ref_logps_from_cache(
                    sample_ids=batch["sample_ids"],
                    ref_logp_cache=ref_logp_cache,
                    device=device,
                    dtype=policy_chosen_logps.dtype,
                )

                # 计算偏好优化损失（DPO / DPOP）
                loss, chosen_reward, rejected_reward, reward_margin, dpop_penalty = _compute_preference_loss(
                    policy_chosen_logps=policy_chosen_logps,
                    policy_rejected_logps=policy_rejected_logps,
                    ref_chosen_logps=ref_chosen_logps,
                    ref_rejected_logps=ref_rejected_logps,
                    beta=args.beta,
                    loss_type=args.loss_type,
                    dpop_lambda=args.dpop_lambda,
                )

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

            global_loss = float(loss.item())
            total_loss.append(global_loss)
            total_reward_margin.append(float(reward_margin.mean().item()))
            total_chosen_reward.append(float(chosen_reward.mean().item()))
            total_rejected_reward.append(float(rejected_reward.mean().item()))
            total_chosen_logp.append(float(policy_chosen_logps.mean().item()))
            total_rejected_logp.append(float(policy_rejected_logps.mean().item()))
            total_dpop_penalty.append(float(dpop_penalty.mean().item()))

            # 打印日志并记录到 TensorBoard
            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                # 计算剩余时间
                rest_time_seconds = spend_time / it * total_iters - spend_time
                rest_time = str(datetime.timedelta(seconds=max(rest_time_seconds, 0)))

                chosen_reward_mean = float(chosen_reward.mean().item())
                rejected_reward_mean = float(rejected_reward.mean().item())
                reward_margin_mean = float(reward_margin.mean().item())
                chosen_logp_mean = float(policy_chosen_logps.mean().item())
                rejected_logp_mean = float(policy_rejected_logps.mean().item())
                dpop_penalty_mean = float(dpop_penalty.mean().item())
                loss_label = args.loss_type.upper()

                log_message = (
                    f"Epoch: {epoch + 1}/{args.epochs} | Step: {step + 1}/{iter_per_epoch} | "
                    f"{loss_label} Loss: {global_loss:.4f} | Reward Margin: {reward_margin_mean:.4f} | "
                    f"Chosen Reward: {chosen_reward_mean:.4f} | Rejected Reward: {rejected_reward_mean:.4f} | "
                    f"Chosen LogP: {chosen_logp_mean:.4f} | Rejected LogP: {rejected_logp_mean:.4f}"
                )
                if args.loss_type == "dpop":
                    log_message += f" | DPOP Penalty: {dpop_penalty_mean:.4f}"
                log_message += f" | LR: {lr:.2e} | s/it: {spend_time / it:.4f} | Remaining time: {rest_time}"
                print(log_message)

                writer.add_scalar(f"Training Loss/{loss_label} Loss", global_loss, it)
                writer.add_scalar("DPO/chosen_reward", chosen_reward_mean, it)
                writer.add_scalar("DPO/rejected_reward", rejected_reward_mean, it)
                writer.add_scalar("DPO/reward_margin", reward_margin_mean, it)
                writer.add_scalar("DPO/chosen_logp", chosen_logp_mean, it)
                writer.add_scalar("DPO/rejected_logp", rejected_logp_mean, it)
                if args.loss_type == "dpop":
                    writer.add_scalar("DPOP/positive_penalty", dpop_penalty_mean, it)
                writer.add_scalar("Learning Rate", lr, it)

    plot_curve_dpo(
        dpo_loss=total_loss,
        reward_margin=total_reward_margin,
        chosen_reward=total_chosen_reward,
        rejected_reward=total_rejected_reward,
        chosen_logp=total_chosen_logp,
        rejected_logp=total_rejected_logp,
        save_path=os.path.join(current_train_path, f"{model_name}_curve.png"),
    )
    print(f"Curve saved to: {os.path.join(current_train_path, f'{model_name}_curve.png')}")
    policy_model.save_pretrained(current_train_path)
    print(f"Model saved to: {current_train_path}")
    writer.close()


# -------------------------------------------【主函数】------------------------------------------- #
def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. GPU is required for training.")
        return

    train_process(args)


if __name__ == "__main__":
    main()
