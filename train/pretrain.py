import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import os
import argparse
import time
import datetime
import math
from pathlib import Path
import json

from mini_models import get_model_and_config, list_models, get_model_info
from data_loader import PreTrainDataset
from utils import (
    setup_ddp,
    cleanup_ddp,
    get_lr,
    configure_optimizer,
    create_folder,
    save_args,
    plot_curve,
    load_checkpoint,
    save_checkpoint,
)


root_path = Path(__file__).parent.parent
tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))
vocab_size = len(tokenizer)


# -------------------------------------------【参数解析】------------------------------------------- #
support_models = ", ".join(list_models())
def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Mini-LLM")

    # 模型与训练精度
    parser.add_argument("--model_name", type=str, required=True, help=f"Mini model names, support: {support_models}")
    parser.add_argument("--precision", type=str, default="bf16", help="Mixed precision training: default bf16, options are fp32 or fp16")

    # 训练参数设置
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum pretraining sequence length")
    parser.add_argument("--max_batch_size", type=int, default=16, help="Training batch size per GPU")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=None, help="Number of warmup iterations")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup iteration ratio")
    parser.add_argument("--lr_decay_iters", type=int, default=None, help="Number of learning rate decay iterations")
    parser.add_argument("--lr_decay_ratio", type=float, default=0.98, help="Learning rate decay iteration ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="Beta parameters for AdamW optimizer")

    # 路径和日志设置
    parser.add_argument("--pretrain_data_path", type=str, default=f"{root_path}/data/pretrain_data/pretrain_data.bin", help="Path to pretrain dataset")
    parser.add_argument("--output_path", type=str, default="./output", help="Model output directory")
    parser.add_argument("--log_interval", type=int, default=50, help="Training log print interval")
    
    # Checkpoint 设置
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from")
    parser.add_argument("--save_interval", type=int, default=None, help="Save checkpoint every N iterations. If None, only save at the end")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep. Older checkpoints will be deleted")

    args = parser.parse_args()

    return args


# -------------------------------------------【训练函数】------------------------------------------- #
def train_process(local_rank, rank, world_size, args):
    # ------------------ 1. DDP设置 ------------------
    is_distributed = world_size > 1
    is_main_process = rank == 0

    # 使用 local_rank 设置当前进程使用的 GPU 设备
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 初始化 DDP
    if is_distributed:
        if is_main_process:  # 仅在主进程打印
            print("Detect DDP training, initializing process group...")
        setup_ddp(rank, world_size)

    # ------------------ 2. 数据准备 ------------------
    dataset = PreTrainDataset(file_path=args.pretrain_data_path, max_seq_len=args.max_seq_len)

    sampler = None
    if is_distributed:
        # DistributedSampler 使用全局 rank 和 world_size
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.max_batch_size,
        pin_memory=True,  # 使用锁页内存提高数据加载速度
        num_workers=4,
        sampler=sampler,
        shuffle=(sampler is None),  # 只有非分布式时才由 DataLoader shuffle
        drop_last=is_distributed,  # 多卡时建议 drop_last
    )

    # --------------- 3. 模型与配置准备 ---------------
    if is_main_process:
        print(f"Support models: {support_models}")
        print(f"Loading model : {args.model_name}")

    Model, Config = get_model_and_config(args.model_name)  # 返回的是模型类和配置类

    # 计算学习率迭代次数参数
    iter_per_epoch = len(dataloader)
    total_iters = args.epochs * iter_per_epoch  # 总迭代次数

    # 预热迭代次数
    if args.warmup_iters is None:
        warmup_iters = int(total_iters * args.warmup_ratio)
    else:
        warmup_iters = args.warmup_iters

    # 衰减迭代次数
    if args.lr_decay_iters is None:
        lr_decay_iters = int(total_iters * args.lr_decay_ratio)
    else:
        lr_decay_iters = args.lr_decay_iters
        assert lr_decay_iters > warmup_iters, "lr_decay_iters must be greater than warmup_iters"
        assert lr_decay_iters <= total_iters, "lr_decay_iters must be less than total_iters"

    # 配置学习率，DDP 模式下，学习率乘以进程数
    # 因为等效 batch 更大，梯度噪声更小
    max_lr = args.max_lr * world_size
    min_lr = args.min_lr * world_size

    # 实例化配置类和模型类
    config = Config(
        vocab_size=vocab_size,
        use_cache=False,
        max_position_embeddings=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # 如果从 checkpoint 恢复，使用 checkpoint 中的配置
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        if is_main_process:
            print(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
        # 从 checkpoint 加载配置
        config_path = os.path.join(args.resume_from_checkpoint, "config.json")
        if os.path.exists(config_path):
            config = Config.from_pretrained(args.resume_from_checkpoint)
            if is_main_process:
                print("Model config loaded from checkpoint")
    
    model = Model(config).to(device)

    # 配置优化器，在 DDP 包装前配置
    optimizer = configure_optimizer(
        model=model,
        weight_decay=args.weight_decay,
        learning_rate=max_lr,
        betas=args.betas,
        device_type="cuda",
    )

    if is_main_process:  # 仅在主进程打印
        print(f"Model info: {json.dumps(get_model_info(model)[1], indent=2)}")
        print(f"Model config: {config}")

    if is_distributed:  # 使用 DDP 包装模型
        # find_unused_parameters=True, DDP 会执行一次额外的检查，识别出哪些参数没有在前向传播中使用，并在梯度同步时跳过它们。
        # 对于 MoE 架构模型，在处理一个 token 时，存在部分未激活的专家，不会接收到梯度，然而，在 batch_size 和 max_seq_len 中存在大量 token
        # 我们设定的专家数较少，几乎可以确定每个专家都会被激活，都会接收到梯度信息，因此这里可以设置为 False
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    
    # 设定混合精度训练，需要在加载 checkpoint 之前初始化 scaler
    scaler = None
    autocast_dtype = None
    enable_amp = False  # 是否启用混合精度训练
    
    if args.precision == "fp16":
        scaler = GradScaler()  # 在 FP16 训练中用于防止梯度下溢的工具
        autocast_dtype = torch.float16
        enable_amp = True
        if is_main_process:
            print("Using FP16 mixed precision training")
    elif args.precision == "bf16":
        autocast_dtype = torch.bfloat16
        enable_amp = True
        if is_main_process:
            print("Using BF16 mixed precision training")
    else:
        if is_main_process:
            print("Using FP32 precision training")
    
    # 从 checkpoint 恢复训练状态，每个 rank 都直接加载相同的 safetensors 文件
    start_epoch = 0
    start_step = 0
    start_iteration = 0
    restored_total_loss = []
    restored_total_ppl = []
    
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        # 所有进程都直接加载 checkpoint
        training_state, training_history = load_checkpoint(
            checkpoint_path=args.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            is_main_process=is_main_process,
        )
        
        if training_state is not None:
            start_epoch = training_state.get("epoch", 0)
            start_step = training_state.get("step", 0)
            start_iteration = training_state.get("iteration", 0)
            if is_main_process:
                print(f"Resuming from epoch {start_epoch}, step {start_step}, iteration {start_iteration}")
        
        # 恢复训练历史数据
        if training_history is not None:
            restored_total_loss = training_history.get("total_loss", [])
            restored_total_ppl = training_history.get("total_ppl", [])
            if is_main_process:
                print(f"Restored training history: {len(restored_total_loss)} loss records, {len(restored_total_ppl)} ppl records")
        
        # 确保所有进程都加载完成后再继续
        if is_distributed:
            dist.barrier()

    # ------------------ 4. 训练循环 ------------------
    if is_main_process:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        model_name = f"pretrained_{args.model_name}"
        
        # 如果从 checkpoint 恢复，使用 checkpoint 的父目录作为输出路径
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            # 从 checkpoint 路径推断输出路径
            # checkpoint 路径格式: output/pretrained_model_name/checkpoints/checkpoint-XXXXX
            checkpoint_parent = os.path.dirname(args.resume_from_checkpoint)  # checkpoints 目录
            current_train_path = os.path.dirname(checkpoint_parent)  # pretrained_model_name 目录
            print(f"Resuming training, using output path: {current_train_path}")
        else:
            current_train_path = os.path.join(args.output_path, model_name)  # ./output/pretrained_model_name
            current_train_path = create_folder(current_train_path)  # 创建文件夹，如果训练了多个模型，则自动添加后缀，例如: pretrained_model_name_1

        # 创建 TensorBoard 日志记录器
        log_dir = os.path.join(current_train_path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)  # ./output/pretrained_model_name/logs

        # 保存本次训练配置
        if not (args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint)):
            save_args(args, os.path.join(current_train_path, f"{model_name}_training_args.json"))
            print(f"Training arguments saved to: {os.path.join(current_train_path, f'{model_name}_training_args.json')}")
        
        # 创建 checkpoint 目录
        checkpoint_dir = os.path.join(current_train_path, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    else:
        # 非主进程也需要知道路径
        model_name = f"pretrained_{args.model_name}"
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            checkpoint_parent = os.path.dirname(args.resume_from_checkpoint)
            current_train_path = os.path.dirname(checkpoint_parent)
        else:
            current_train_path = os.path.join(args.output_path, model_name)
        checkpoint_dir = os.path.join(current_train_path, "checkpoints")
    
    # 用于跟踪已保存的 checkpoint
    saved_checkpoints = []

    # 初始化训练历史数据（如果从 checkpoint 恢复，使用恢复的数据）
    total_loss = restored_total_loss.copy() if restored_total_loss else []
    total_ppl = restored_total_ppl.copy() if restored_total_ppl else []
    start_time = time.time()

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        if is_distributed and hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)  # 让每一个 epoch 的数据 shuffle 使用不同的随机种子
        model.train()  # 设置模型为训练模式

        # 如果从 checkpoint 恢复，需要跳过已经训练过的 step
        start_step_in_epoch = start_step if epoch == start_epoch else 0
        
        for step, input_ids in enumerate(dataloader):
            # 跳过已经训练过的 step
            if epoch == start_epoch and step < start_step_in_epoch:
                continue
            input_ids = input_ids.to(device)

            # 清零梯度
            optimizer.zero_grad(set_to_none=True)

            it = epoch * iter_per_epoch + step + 1  # 当前全局迭代次数
            lr = get_lr(
                it,
                max_lr=max_lr,
                min_lr=min_lr,
                warmup_iters=warmup_iters,
                lr_decay_iters=lr_decay_iters,
            )  # 获取当前学习率
            for param_group in optimizer.param_groups:  # 将新的学习率值应用到优化器中
                param_group["lr"] = lr

            with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):  # 使用混合精度训练
                # NOTE: 重要！！ transformers 的 loss_function 会在内部对 label 进行 shift 操作
                # 因此这里传入的 labels 实际上就是 input_ids 本身，transformers 内部会自动补全一个 ignore_index 并进行 shift
                # 详见 transformer.loss.loss_utils.py 的 ForCausalLMLoss
                # 如果模型需要自定义实现 loss，则需要在自定义 loss 中实现 shift 操作，从而兼容当前的训练代码
                # 因此，相比 Mini-LLM V1，dataloader 不再返回 shift 的 labels
                outputs: CausalLMOutputWithPast = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

            if scaler is not None:  # 意味着正在使用 FP16
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # FP32 或 BF16
                # DDP会自动对每个卡的loss求梯度，然后对梯度求平均
                # 由于在预训练时，不涉及到pad、ignore_index等无效token，因此每个卡直接求梯度即可
                # 如果每个卡的有效token数不一致，需要在每个卡上进行归一化，即除以全局有效token数
                loss.backward()
                optimizer.step()

            if is_distributed:  # 如果是 DDP，记录全局损失
                reduced_loss = loss.clone().detach().to(device)
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                global_loss = reduced_loss.item() / world_size
            else:
                global_loss = loss.item()
            total_loss.append(global_loss)
            ppl = math.exp(global_loss)
            total_ppl.append(ppl)

            # 确保所有进程完成该轮的计算
            if is_distributed:
                dist.barrier()  # 同步所有进程

            # 打印日志并记录到 TensorBoard
            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                # 计算剩余时间
                rest_time = spend_time / it * total_iters - spend_time
                rest_time = str(datetime.timedelta(seconds=rest_time))
                if is_main_process:
                    print(f"Epoch: {epoch + 1}/{args.epochs} | Step: {step + 1}/{iter_per_epoch} | Global Loss: {global_loss:.4f} | PPL: {ppl:.4f} | LR: {lr:.6f} | Seconds/Iteration: {spend_time / it:.4f} | Remaining time: {rest_time}")
                    # 记录到 TensorBoard
                    writer.add_scalar("Training Loss/Global Loss", global_loss, it)
                    writer.add_scalar("Learning Rate", lr, it)
                    writer.add_scalar("Perplexity", ppl, it)
                    # 对特定模型额外记录序列级辅助损失、mtp 损失和负载情况等
                    if args.model_name == 'mini_deepseekv3' or args.model_name == 'mini_qwen3_next':
                        if hasattr(outputs, 'total_seq_aux_loss') and outputs.total_seq_aux_loss:
                            writer.add_scalar('Training Loss/Seq Aux Loss', outputs.total_seq_aux_loss, it)
                        if hasattr(outputs, 'total_mtp_loss') and outputs.total_mtp_loss:
                            writer.add_scalar('Training Loss/MTP Loss', outputs.total_mtp_loss, it)
                        if hasattr(outputs, 'aux_loss') and outputs.aux_loss:
                            writer.add_scalar('Training Loss/Aux Loss', outputs.aux_loss, it)
                        if hasattr(outputs, 'all_global_counts') and outputs.all_global_counts:
                            for layer_info in outputs.all_global_counts:
                                layer_idx = layer_info['layer_idx']
                                counts = torch.tensor(layer_info['global_counts'], dtype=torch.float32)
                                max_load = counts.max()
                                min_load = counts.min()
                                ratio = (max_load / (min_load + 1e-6)).item()
                                writer.add_scalar(f"ExpertBalance/Layer_{layer_idx}/max_min_ratio", ratio, it)  # 记录每层专家负载的 max/min 比例，趋近于 1 表示负载均衡，越大表示负载不均衡
            
            # 保存 checkpoint
            if args.save_interval is not None and it % args.save_interval == 0:
                if is_distributed:
                    dist.barrier()  # 确保所有进程同步
                saved_checkpoints = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    iteration=it,
                    lr=lr,
                    total_loss=total_loss,
                    total_ppl=total_ppl,
                    checkpoint_dir=checkpoint_dir,
                    saved_checkpoints=saved_checkpoints,
                    save_total_limit=args.save_total_limit,
                    is_main_process=is_main_process,
                )

    if is_main_process:
        # 绘制损失曲线
        plot_curve(total_loss, total_ppl, os.path.join(current_train_path, f"{model_name}_curve.png"))
        print(f"Curve saved to: {os.path.join(current_train_path, f'{model_name}_curve.png')}")
        # 保存模型时解开 DDP 包装
        model_to_save = model.module if hasattr(model, "module") else model
        
        # 训练完成后，删除 MTP 模块
        if args.model_name == 'mini_deepseekv3':
            if hasattr(model_to_save, 'remove_mtp_module') and model_to_save.mtp is not None:
                print("Removing MTP module before final save...")
                model_to_save.remove_mtp_module()
                print("MTP module removed successfully")
        
        model_to_save.save_pretrained(current_train_path)
        print(f"Model saved to: {current_train_path}")

    if is_main_process:
        writer.close()  # 关闭 TensorBoard 日志记录器

    cleanup_ddp()  # 清理 DDP 环境


# -------------------------------------------【主函数】------------------------------------------- #
def main():
    # 参数解析
    args = parse_args()

    # 使用 .get() 为环境变量提供默认值，使用 torchrun 时会覆盖这些环境变量，若未使用 torchrun 则兼容单卡训练
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1 and not dist.is_available():
        print("DDP training detected, but the current environment does not support it.")
        return

    if not torch.cuda.is_available():
        print(f"RANK-{rank} LOCAL_RANK-{local_rank}: CUDA is not available. GPU is required for training.")
        return

    if local_rank >= torch.cuda.device_count():
        print(f"RANK-{rank} LOCAL_RANK-{local_rank}: Please check that torchrun's --nproc_per_node parameter is less than or equal to {torch.cuda.device_count()}")
        return

    # 每个进程直接调用训练函数
    train_process(local_rank, rank, world_size, args)


if __name__ == "__main__":
    main()
