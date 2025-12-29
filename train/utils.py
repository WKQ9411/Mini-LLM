import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import inspect
from typing import Tuple, Optional, List
from transformers import PreTrainedModel
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import shutil
import glob


# ------------------------------【DDP (Distributed Data Parallel) 相关函数】--------------------------- #
def setup_ddp(rank, world_size):
    """初始化分布式环境。"""
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """销毁进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------------------------【训练相关函数】------------------------------------------- #
def get_lr(it, max_lr, min_lr, warmup_iters, lr_decay_iters):
    """
    根据迭代次数返回学习率, it为总迭代次数
    """
    # 1. warmup 阶段
    if it < warmup_iters:
        return max_lr * it / warmup_iters  # 线性增加到最大学习率
    # 2. 衰减结束，使用最小学习率
    if it > lr_decay_iters:
        return min_lr  # 衰减结束，使用最小学习率
    # 3. 余弦衰减阶段
    decay_ratio = (it - warmup_iters) / (
        lr_decay_iters - warmup_iters
    )  # 衰减阶段中，当前迭代相对于剩余迭代的比例
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff 是一个从 0 到 1 之间变化的系数，控制学习率的衰减

    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(
    model: PreTrainedModel,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str = "cuda",
):
    """
    配置 AdamW 优化器, 并对参数分组, 以应用不同的优化策略, 通常权重矩阵(2D及以上)应用权重衰减, 而偏置(bias)和层归一化(LayerNorm)的参数不应用权重衰减

    Args:
        model (PreTrainedModel): 模型
        weight_decay (float): 权重衰减系数
        learning_rate (float): 学习率
        betas (Tuple[float, float]): AdamW 优化器的 beta1 和 beta2 参数
        device_type (str): 设备类型, 用于指定优化器的设备

    Returns:
        torch.optim.AdamW: 优化器
    """
    # 获取模型参数并过滤不需要梯度的参数
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # 维度大于等于 2 的参数（如权重矩阵、嵌入层参数），这些参数会应用权重衰减
    # 这些参数通常是模型的主要可学习参数，直接影响模型的表达能力
    # 维度小于 2 的参数（如偏置、LayerNorm 参数），这些参数不会应用权重衰减
    # 这些参数通常用于调整模型的输出分布，而不是直接参与特征变换
    decay_params = []
    no_decay_params = []

    for name, param in param_dict.items():
        if param.dim() < 2 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # 创建优化器参数组
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # 检查是否支持融合 AdamW
    # 融合 AdamW（Fused AdamW） 是 PyTorch 提供的一种优化 AdamW 实现的高性能版本，通过将多个操作融合为一个内核（kernel）来加速计算
    # 它特别适用于 GPU 上的大规模深度学习训练任务
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )

    return optimizer


# -------------------------------------------【Checkpoint 相关函数】------------------------------------------- #
def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    step,
    iteration,
    lr,
    total_loss,
    total_ppl,
    checkpoint_dir,
    saved_checkpoints,
    save_total_limit,
    is_main_process,
):
    """
    保存训练 checkpoint
    
    Args:
        model: 模型, 可能是 DDP 包装的
        optimizer: 优化器
        scaler: GradScaler, 如果使用 FP16
        epoch: 当前 epoch
        step: 当前 step
        iteration: 全局迭代次数
        lr: 当前学习率
        total_loss: 训练损失历史列表
        total_ppl: 困惑度历史列表
        checkpoint_dir: checkpoint 保存目录
        saved_checkpoints: 已保存的 checkpoint 列表
        save_total_limit: 最多保留的 checkpoint 数量
        is_main_process: 是否为主进程
    
    Returns:
        list: 更新后的 saved_checkpoints 列表
    """
    if not is_main_process:
        return saved_checkpoints
    
    # 创建 checkpoint 路径
    checkpoint_name = f"checkpoint-{iteration}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # 保存模型
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(checkpoint_path)
    
    # 保存优化器状态
    optimizer_state_path = os.path.join(checkpoint_path, "optimizer.pt")
    torch.save(optimizer.state_dict(), optimizer_state_path)
    
    # 保存 scaler 状态
    if scaler is not None:
        scaler_state_path = os.path.join(checkpoint_path, "scaler.pt")
        torch.save(scaler.state_dict(), scaler_state_path)
    
    # 保存训练状态
    training_state = {
        "epoch": epoch,
        "step": step,
        "iteration": iteration,
        "learning_rate": lr,
    }
    training_state_path = os.path.join(checkpoint_path, "training_state.json")
    with open(training_state_path, "w", encoding="utf-8") as f:
        json.dump(training_state, f, indent=2, ensure_ascii=False)
    
    # 保存训练历史数据
    training_history = {
        "total_loss": total_loss,
        "total_ppl": total_ppl,
    }
    training_history_path = os.path.join(checkpoint_path, "training_history.json")
    with open(training_history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    # 添加到已保存列表
    saved_checkpoints.append(checkpoint_path)
    
    # 如果超过限制，删除最旧的 checkpoint
    if len(saved_checkpoints) > save_total_limit:
        oldest_checkpoint = saved_checkpoints.pop(0)
        if os.path.exists(oldest_checkpoint):
            shutil.rmtree(oldest_checkpoint)
            print(f"Deleted old checkpoint: {oldest_checkpoint}")
    
    print(f"Checkpoint saved to: {checkpoint_path}")
    return saved_checkpoints


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scaler,
    device,
    is_main_process=False,
):
    """
    从 checkpoint 加载训练状态，每个 rank 直接加载权重文件
    
    Args:
        checkpoint_path: checkpoint 目录路径
        model: 模型, 可能是 DDP 包装的
        optimizer: 优化器
        scaler: GradScaler, 如果使用 FP16
        device: 当前设备
        is_main_process: 是否为主进程
    
    Returns:
        tuple: (training_state, training_history)
            - training_state: 包含 epoch, step, iteration, learning_rate 的训练状态
            - training_history: 包含 total_loss 和 total_ppl 的训练历史数据
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    if is_main_process:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 获取实际的模型
    model_to_load = model.module if hasattr(model, "module") else model
    
    # 加载 safetensors 权重文件, 支持单个文件和分片文件
    from safetensors.torch import load_file
    
    # 查找所有 safetensors 文件
    safetensors_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in checkpoint directory: {checkpoint_path}")
    
    # 优先使用单个 model.safetensors 文件
    single_file = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(single_file):
        # 单个文件情况
        state_dict = load_file(single_file, device=device)
        if is_main_process:
            print(f"Model weights loaded from model.safetensors on {device}")
    else:
        # 分片文件情况
        safetensors_files.sort()
        state_dict = {}
        for safetensors_file in safetensors_files:
            file_state_dict = load_file(safetensors_file, device=device)
            state_dict.update(file_state_dict)
        if is_main_process:
            print(f"Model weights loaded from {len(safetensors_files)} sharded safetensors files on {device}")
    
    model_to_load.load_state_dict(state_dict, strict=False)
    
    # 加载优化器状态
    optimizer_state_path = os.path.join(checkpoint_path, "optimizer.pt")
    if os.path.exists(optimizer_state_path):
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
        if is_main_process:
            print("Optimizer state loaded")
    
    # 加载 scaler 状态
    if scaler is not None:
        scaler_state_path = os.path.join(checkpoint_path, "scaler.pt")
        if os.path.exists(scaler_state_path):
            scaler.load_state_dict(torch.load(scaler_state_path, map_location=device))
            if is_main_process:
                print("Scaler state loaded")
    
    # 加载训练状态
    training_state = None
    training_state_path = os.path.join(checkpoint_path, "training_state.json")
    if os.path.exists(training_state_path):
        with open(training_state_path, "r", encoding="utf-8") as f:
            training_state = json.load(f)
        if is_main_process:
            print(f"Training state loaded: epoch={training_state.get('epoch')}, step={training_state.get('step')}, iteration={training_state.get('iteration')}")
    else:
        if is_main_process:
            print("Warning: training_state.json not found, returning None")
    
    # 加载训练历史数据
    training_history = None
    training_history_path = os.path.join(checkpoint_path, "training_history.json")
    if os.path.exists(training_history_path):
        with open(training_history_path, "r", encoding="utf-8") as f:
            training_history = json.load(f)
        if is_main_process:
            print(f"Training history loaded: {len(training_history.get('total_loss', []))} loss records, {len(training_history.get('total_ppl', []))} ppl records")
    
    return training_state, training_history


# ---------------------------------------------【工具函数】--------------------------------------------- #
def create_folder(base_path):
    """
    创建文件夹，如果文件夹已存在，则在文件夹名称后添加数字后缀，返回添加了后缀的文件夹路径
    """
    folder_name = base_path
    counter = 1
    # 如果文件夹存在，尝试添加尾号
    while os.path.exists(folder_name):
        folder_name = f"{base_path}_{counter}"
        counter += 1
    # 创建文件夹
    os.makedirs(folder_name)
    return folder_name


def save_args(args, save_path):
    """
    保存本次训练的命令行参数, 供实验参考

    Args:
        args: 解析的命令行参数对象
        save_path: JSON文件路径, 用于保存参数
    """
    # 将args对象转换为字典
    args_dict = vars(args)

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将参数保存为JSON文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)


def str2bool(v):
    """
    将字符串转换为布尔值, 支持 "true", "false", "1", "0" 等多种格式
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "t", "y", "yes"):
        return True
    elif v.lower() in ("false", "0", "f", "n", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_args(args_path):
    """
    从 json 文件加载训练参数

    Args:
        args_path (str): json 文件路径, 用于加载参数
    """
    # 从 json 文件加载参数
    with open(args_path, "r", encoding="utf-8") as f:
        args_dict = json.load(f)

    return args_dict


def plot_curve(total_loss, total_ppl=None, save_path=None):
    """
    绘制训练损失、困惑度曲线并保存

    Args:
        total_loss (list): 训练损失列表
        total_ppl (list or None): 困惑度列表, 如果为None则只绘制损失曲线
        save_path (str or None): 保存路径, 如果为None则不保存图片
    """

    # 设置美化样式
    plt.style.use("seaborn-v0_8")
    
    # 根据是否有PPL数据决定创建一个还是两个子图
    if total_ppl is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 生成迭代索引
    iterations = range(len(total_loss))

    # ====== 损失曲线 ======
    # 原始曲线（淡化）
    ax1.plot(
        iterations,
        total_loss,
        alpha=0.3,
        color="lightblue",
        linewidth=1,
        label="Original Loss",
    )

    # 拟合曲线（深色）- 使用移动平均
    window_size = min(50, len(total_loss) // 10) if len(total_loss) > 10 else 1
    if window_size > 1:
        smoothed_loss = np.convolve(total_loss, np.ones(window_size) / window_size, mode="valid")
        smooth_iterations = range(window_size // 2, len(total_loss) - window_size // 2 + 1)
        ax1.plot(
            smooth_iterations,
            smoothed_loss,
            color="darkblue",
            linewidth=2.5,
            label="Smoothed Loss",
        )
    else:
        ax1.plot(iterations, total_loss, color="darkblue", linewidth=2.5, label="Loss")

    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_facecolor("#f8f9fa")

    # ====== 困惑度曲线 ======
    if total_ppl is not None:
        # 原始曲线（淡化）
        ax2.plot(
            iterations,
            total_ppl,
            alpha=0.3,
            color="lightcoral",
            linewidth=1,
            label="Original PPL",
        )

        # 拟合曲线（深色）- 使用移动平均
        window_size = min(50, len(total_ppl) // 10) if len(total_ppl) > 10 else 1
        if window_size > 1:
            smoothed_ppl = np.convolve(total_ppl, np.ones(window_size) / window_size, mode="valid")
            smooth_iterations = range(window_size // 2, len(total_ppl) - window_size // 2 + 1)
            ax2.plot(
                smooth_iterations,
                smoothed_ppl,
                color="darkred",
                linewidth=2.5,
                label="Smoothed PPL",
            )
        else:
            ax2.plot(iterations, total_ppl, color="darkred", linewidth=2.5, label="PPL")

        ax2.set_xlabel("Iterations", fontsize=12)
        ax2.set_ylabel("Perplexity", fontsize=12)
        ax2.set_title("Perplexity Curve", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_facecolor("#f8f9fa")

    # 整体美化
    plt.tight_layout()
    fig.patch.set_facecolor("white")

    # 保存图片
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
