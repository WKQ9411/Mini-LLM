import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os
import argparse
import time
import datetime
import math
from model import list_models, get_model_and_args
from utils.little_tools import create_folder, plot_loss_curve, save_to_yaml
from datasets import PreTrainDataset
import json
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("./mini_tokenizer")
vocab_size = len(tokenizer)

# -------------------------------------------【参数解析】------------------------------------------- #
support_models = "、".join(list_models())
def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Mini Language Model")

    # 模型与训练精度
    parser.add_argument("--model_name", type=str, required=True, help=f"预训练模型名称，当前支持的模型有：{support_models}")
    parser.add_argument("--precision", type=str, default='bf16', help="选择使用混合精度训练: 默认 bf16, 可选 fp32 或 fp16")

    # ModelArgs 设置
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--max_batch_size", type=int, default=16, help="每个GPU的训练批次大小")
    parser.add_argument("--epochs", type=int, default=1, help="训练周期")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="最大学习率")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率")
    parser.add_argument("--warmup_iters", type=int, default=None, help="预热迭代次数")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="预热迭代比例")
    parser.add_argument("--lr_decay_iters", type=int, default=None, help="学习率衰减迭代次数")
    parser.add_argument("--lr_decay_ratio", type=float, default=0.98, help="学习率衰减迭代比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减系数")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="AdamW 优化器的 beta 参数")
    
    # 路径与日志设置
    parser.add_argument("--pretrain_data_path", type=str, default='./preprocess_data/data/pretrain_data/pretrain_data.bin', help="预训练数据集路径")
    parser.add_argument("--output_path", type=str, default="./output", help="模型保存目录")
    parser.add_argument("--log_interval", type=int, default=10, help="训练日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=50000, help="模型保存间隔")

    args = parser.parse_args()

    return args


# -------------------------------------------【DDP 函数】------------------------------------------- #
def setup_ddp(rank, world_size):
    """初始化分布式环境。"""
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    """销毁进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------------------------【学习率调度】------------------------------------------- #
def get_lr(it, model_args):
    """
    根据迭代次数返回学习率, it为总迭代次数
    """
    max_lr = model_args.max_lr  # 最大学习率
    min_lr = model_args.min_lr  # 最小学习率
    warmup_iters = model_args.warmup_iters  # 预热迭代次数
    lr_decay_iters = model_args.lr_decay_iters  # 衰减迭代次数

    # 1. warmup 阶段
    if it < warmup_iters:
        return max_lr * it / warmup_iters  # 线性增加到最大学习率
    # 2. 衰减结束，使用最小学习率
    if it > lr_decay_iters:
        return min_lr  # 衰减结束，使用最小学习率
    # 3. 余弦衰减阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # 衰减阶段中，当前迭代相对于剩余迭代的比例
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff 是一个从 0 到 1 之间变化的系数，控制学习率的衰减
    return min_lr + coeff * (max_lr - min_lr)


# -------------------------------------------【训练函数】------------------------------------------- #
def train_process(local_rank, rank, world_size, args):

    # ------------------ 1. DDP设置 ------------------
    is_distributed = world_size > 1
    is_main_process = (rank == 0)

    # 使用 local_rank 设置当前进程使用的 GPU 设备
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 初始化 DDP
    if is_distributed:
        if is_main_process:  # 仅在主进程打印
            print("检测到 DDP 训练，初始化进程组...")
        setup_ddp(rank, world_size)

    # ------------------ 2. 数据准备 ------------------
    dataset = PreTrainDataset(
        file_path=args.pretrain_data_path,
        max_seq_len=args.max_seq_len
        )
    
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
        drop_last=is_distributed  # 多卡时建议 drop_last
    )

    # --------------- 3. 模型与配置准备 --------------- 
    if is_main_process:
        print(f"当前支持的模型有: {support_models}")
        print(f"当前加载的模型为: {args.model_name}")

    Model, Model_Args = get_model_and_args(args.model_name)  # 返回的是模型类和配置类

    # 计算学习率迭代次数参数
    iter_per_epoch = len(dataloader)
    total_iters = args.epochs * iter_per_epoch  # 总迭代次数
    if args.warmup_iters is None:
        warmup_iters = int(total_iters * args.warmup_ratio)  # 预热迭代次数
    else:
        warmup_iters = args.warmup_iters
    if args.lr_decay_iters is None:
        lr_decay_iters = int(total_iters * args.lr_decay_ratio)  # 衰减迭代次数
    else:
        lr_decay_iters = args.lr_decay_iters
        assert lr_decay_iters > warmup_iters, "衰减迭代次数必须大于预热迭代次数"
        assert lr_decay_iters <= total_iters, "衰减迭代次数必须小于总迭代次数"

    # 实例化配置类，将命令行参数传入模型配置
    model_args = Model_Args(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size, 
        epochs=args.epochs,
        max_lr=args.max_lr * world_size,  # DDP 模式下，学习率乘以进程数
        min_lr=args.min_lr * world_size, 
        warmup_iters=warmup_iters,
        warmup_ratio=args.warmup_ratio,
        lr_decay_iters=lr_decay_iters,
        lr_decay_ratio=args.lr_decay_ratio,
        weight_decay=args.weight_decay,
        betas=args.betas,
        )
    model = Model(model_args).to(device)  # 实例化模型类，传入配置

    # 配置优化器，在 DDP 包装前配置
    optimizer = model.configure_optimizer(
        weight_decay=model_args.weight_decay,
        learning_rate=model_args.max_lr,
        betas=model_args.betas,
        device_type='cuda'
    )

    if is_main_process:  # 仅在主进程打印
        print(f"模型参数量: {model.count_parameters()[1]}")
        print(f"模型配置: {json.dumps(asdict(model_args), indent=4)}")
    
    if is_distributed:  # 使用 DDP 包装模型
        # find_unused_parameters=True, DDP 会执行一次额外的检查，识别出哪些参数没有在前向传播中使用，并在梯度同步时跳过它们。
        # 对于 MoE 架构模型，在处理一个 token 时，存在部分未激活的专家，不会接收到梯度，然而，在 batch_size 和 max_seq_len 中存在大量 token
        # 我们设定的专家数较少，几乎可以确定每个专家都会被激活，都会接收到梯度信息，因此这里可以设置为 False
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # ------------------ 4. 训练循环 ------------------
    if is_main_process:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        model_name = f'pretrained_{args.model_name}'
        current_train_path = os.path.join(args.output_path, model_name)  # ./output/pretrained_model_name
        current_train_path = create_folder(current_train_path)  # 创建文件夹，如果训练了多个模型，则自动添加后缀，例如: pretrained_model_name_1
        
        # 创建 TensorBoard 日志记录器
        log_dir = os.path.join(current_train_path, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)  # ./output/pretrained_model_name/logs

        # 保存本次训练配置
        save_to_yaml(asdict(model_args), os.path.join(current_train_path, f'{model_name}_model_args.yaml'))
        print(f"本次训练配置保存至: {os.path.join(current_train_path, f'{model_name}_model_args.yaml')}")

    total_loss = []
    start_time = time.time()
    scaler = None
    autocast_dtype = None
    enable_amp = False

    if args.precision == 'fp16':
        scaler = GradScaler()
        autocast_dtype = torch.float16
        enable_amp = True
        if is_main_process:
            print("使用 FP16 混合精度训练")
    elif args.precision == 'bf16':
        autocast_dtype = torch.bfloat16
        enable_amp = True
        if is_main_process:
            print("使用 BF16 混合精度训练")
    else:
        if is_main_process:
            print("使用 FP32 精度训练")

    for epoch in range(model_args.epochs):

        if is_distributed and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)  # 设置 sampler 的 epoch 以保证 shuffle 正确
        model.train()  # 设置模型为训练模式

        for step, (X, Y) in enumerate(dataloader):
            
            X, Y = X.to(device), Y.to(device)

            # 清零梯度
            optimizer.zero_grad(set_to_none=True)

            it = epoch * iter_per_epoch + step + 1  # 当前全局迭代次数
            lr = get_lr(it, model_args)  # 获取当前学习率
            for param_group in optimizer.param_groups:  # 将新的学习率值应用到优化器中
                param_group['lr'] = lr

            with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):  # 使用混合精度训练
                _, loss, other_vars = model(X, Y)
            
            if scaler is not None: # 意味着正在使用 FP16
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # FP32 或 BF16
                loss.backward()
                optimizer.step()

            if is_distributed:  # 如果是 DDP，记录全局损失
                reduced_loss = loss.clone().detach().to(device)
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                global_loss = reduced_loss.item() / world_size
            else:
                global_loss = loss.item()
            total_loss.append(global_loss)

            # 确保所有进程完成该轮的计算
            if is_distributed:
                dist.barrier()  # 同步所有进程

            if is_main_process:
                if it % args.save_interval == 0:
                    # 保存模型时解开 DDP 包装
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_path = os.path.join(current_train_path, f'{model_name}_epoch_{epoch+1}_iter_{it}_loss_{global_loss:.2f}-base.pt')
                    # 原子化保存（防写入中断）
                    tmp_path = model_path + ".tmp"
                    torch.save(model_to_save.state_dict(), tmp_path)
                    os.rename(tmp_path, model_path)  # 重命名为最终文件名
                    print(f"模型保存至: {model_path}")

            # 打印日志
            if step != 0 and step % args.log_interval == 0:
                spend_time = time.time() - start_time
                # 计算剩余时间
                rest_time = spend_time / it * total_iters - spend_time
                rest_time = str(datetime.timedelta(seconds=rest_time))
                if is_main_process:
                    print(f"Epoch: {epoch+1}/{model_args.epochs} | Step: {step+1}/{iter_per_epoch} | Global Loss: {global_loss:.4f} | LR: {lr:.6f} | 预计剩余训练时间: {rest_time}")
            if is_main_process:
                writer.add_scalar('Training Loss', global_loss, it)
                writer.add_scalar('Learning Rate', lr, it)
                if args.model_name == 'mini_deepseekv3':  # 对 mini_deepseekv3 模型记录具体的损失值和负载情况
                    if other_vars[0] != 0.0:
                        writer.add_scalar('Main Loss', other_vars[0], it)
                    if other_vars[1] != 0.0:
                        writer.add_scalar('Seq Aux Loss', other_vars[1], it)
                    if other_vars[2] != 0.0:
                        writer.add_scalar('MTP Loss', other_vars[2], it)
                    for i, counts in enumerate(other_vars[3], start=model_args.n_dense_layers):
                        writer.add_histogram(f'Layer {i+1} Load', counts, it)
    
    if is_main_process:
        # 绘制损失曲线
        plot_loss_curve(total_loss, os.path.join(current_train_path, f'{model_name}_loss_curve.png'))
        print(f"损失曲线保存至: {os.path.join(current_train_path, f'{model_name}_loss_curve.png')}")
        # 保存模型时解开 DDP 包装
        model_to_save = model.module if hasattr(model, 'module') else model
        model_path = os.path.join(current_train_path, f'{model_name}_final_loss_{global_loss:.2f}-base.pt')
        # 原子化保存（防写入中断）
        tmp_path = model_path + ".tmp"
        torch.save(model_to_save.state_dict(), tmp_path)
        os.rename(tmp_path, model_path)  # 重命名为最终文件名
        print(f"模型保存至: {model_path}")
    
    if is_main_process:
        writer.close()  # 关闭 TensorBoard 日志记录器
    
    cleanup_ddp()  # 清理 DDP 环境


# -------------------------------------------【主函数】------------------------------------------- #
def main():
    # 参数解析
    args = parse_args()

    # 使用 .get() 为环境变量提供默认值，使用 torchrun 时会覆盖这些值，若未使用 torchrun 则兼容单卡训练
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1 and not dist.is_available():
        print("检测到使用 DDP 训练，但当前环境不支持。")
        return

    if not torch.cuda.is_available():
         print(f"RANK-{rank} LOCAL_RANK-{local_rank}: CUDA 不可用，需要使用 GPU 进行训练。")
         return

    if local_rank >= torch.cuda.device_count():
        print(f"RANK-{rank} LOCAL_RANK-{local_rank}: 请检查 torchrun 的 --nproc_per_node 参数是否小于等于 {torch.cuda.device_count()}")
        return

    # 每个进程直接调用训练函数
    train_process(local_rank, rank, world_size, args)


if __name__ == '__main__':
    main()



