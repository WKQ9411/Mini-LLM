import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

import os
import argparse
import time
import datetime
from pathlib import Path
import json

from mini_models import get_model_and_config, list_models, get_model_info
from data_loader import SFTDataset
from utils import (
    get_lr,
    configure_optimizer,
    create_folder,
    save_args,
    load_args,
    plot_curve,
)


root_path = Path(__file__).parent.parent
tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))
vocab_size = len(tokenizer)
pad_id = tokenizer.pad_token_id  # <|endoftext|>


# -------------------------------------------【参数解析】------------------------------------------- #
support_models = ", ".join(list_models())
def parse_args():
    parser = argparse.ArgumentParser(description="SFT Mini-LLM")

    # 模型与训练精度
    parser.add_argument("--model_name", type=str, required=True, help=f"Mini model names, support: {support_models}")
    parser.add_argument("--precision", type=str, default='bf16', help="Mixed precision training: default bf16, options are fp32 or fp16")
    parser.add_argument("--base_model_path", type=str, default=f"{root_path}/output/pretrained_mini_deepseekv3", help="Base model checkpoint path")

    # ModelArgs 设置
    parser.add_argument("--max_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=None, help="Number of warmup iterations")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup iteration ratio")
    parser.add_argument("--lr_decay_iters", type=int, default=None, help="Number of learning rate decay iterations")
    parser.add_argument("--lr_decay_ratio", type=float, default=0.98, help="Learning rate decay iteration ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="Beta parameters for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    
    # 路径与日志设置
    parser.add_argument("--sft_data_path", type=str, default=f"{root_path}/data/sft_data/parquet/packed_sft_data.parquet", help="Path to sft dataset")
    parser.add_argument("--output_path", type=str, default="./output", help="Model output directory")
    parser.add_argument("--log_interval", type=int, default=50, help="Training log print interval")

    args = parser.parse_args()

    return args


# -------------------------------------------【训练函数】------------------------------------------- #
def train_process(args):

    # ------------------ 1. 设置 ------------------
    # 设置当前使用的设备
    device = torch.device("cuda")

    # 获取预训练模型的训练配置
    model_args_path = f"{args.base_model_path}/pretrained_{args.model_name}_training_args.json"
    base_model_training_args = load_args(model_args_path)
    args.max_seq_len = base_model_training_args['max_seq_len']

    # ------------------ 2. 数据准备 ------------------
    dataset = SFTDataset(
        file_path=args.sft_data_path,
        max_seq_len=args.max_seq_len,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.max_batch_size,
        shuffle=True,
    )

    iter_per_epoch = len(dataloader)
    total_iters = args.epochs * iter_per_epoch  # 总迭代次数

    # --------------- 3. 模型与配置准备 --------------- 
    print(f"Support models: {support_models}")
    print(f"Loading model : {args.model_name}")

    Model, Config = get_model_and_config(args.model_name)  # 返回的是模型类和配置类
    model = Model.from_pretrained(args.base_model_path).to(device)
    print(f"Loading pretrained model weights from: {args.base_model_path}")

    # 特定模型配置调整
    if args.model_name == 'mini_deepseekv3':
        # 当前在 mini_deepseekv3 模型的实现中，router、seq_aux 暂未考虑 pad token，集中的 pad token 可能影响专家负载
        # 这里简单的禁用 noaux_load_balance 和 seq_aux，并推荐使用 packing 模式的 sft 数据集减少 pad token 的影响
        # mtp 在预训练时已经引导主模型具备了一定的预测未来多个 token 的能力，这里为了简化直接禁用
        model.config.use_mtp = False
        model.config.use_noaux_load_balance = False
        model.config.use_seq_aux = False

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
    
    # 配置优化器
    optimizer = configure_optimizer(
        model=model,
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        betas=args.betas,
        device_type="cuda",
    )

    print(f"Model info: {json.dumps(get_model_info(model)[1], indent=2)}")
    # print(f"Model config: {json.dumps(model.config.to_dict(), indent=2)}")
    
    # ------------------ 4. 训练循环 ------------------
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    model_name = f'sft_{args.model_name}'
    current_train_path = os.path.join(args.output_path, model_name)  # ./output/sft_model_name
    current_train_path = create_folder(current_train_path)  # 创建文件夹，如果训练了多个模型，则自动添加后缀，例如: sft_model_name_1
        
    # 创建 TensorBoard 日志记录器
    log_dir = os.path.join(current_train_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)  # ./output/sft_model_name/logs

    # 保存本次训练配置
    save_args(args, os.path.join(current_train_path, f"{model_name}_training_args.json"))
    print(f"Training arguments saved to: {os.path.join(current_train_path, f'{model_name}_training_args.json')}")

    total_loss = []
    start_time = time.time()
    scaler = None
    autocast_dtype = None
    enable_amp = False

    # 设定混合精度训练
    if args.precision == 'fp16':
        scaler = GradScaler()
        autocast_dtype = torch.float16
        enable_amp = True
        print("Using FP16 mixed precision training")
    elif args.precision == 'bf16':
        autocast_dtype = torch.bfloat16
        enable_amp = True
        print("Using BF16 mixed precision training")
    else:
        print("Using FP32 precision training")

    for epoch in range(args.epochs):
        model.train()  # 设置模型为训练模式
        for step, data in enumerate(dataloader):
            
            input_ids, labels = data["input_ids"].to(device), data["labels"].to(device)
            position_ids, attention_mask = data["position_ids"].to(device), data["attention_mask"].to(device)

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
                # transformers 的 loss_function 会在内部对 label 进行 shift 操作
                # 当前传入的 labels 是与 input_ids 对齐的，会在内部进行 shift 操作
                outputs: CausalLMOutputWithPast = model(
                    input_ids=input_ids, 
                    position_ids=position_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                    )  # 由于 SFTDataset 设置了默认的 ignore_index=-100，内部 loss 能够直接计算需生成部分的 loss
                loss = outputs.loss
            
            if scaler is not None: # 意味着正在使用 FP16
                scaler.scale(loss).backward()
                # 梯度裁剪，需要先 unscale，然后裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else: # FP32 或 BF16
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            global_loss = loss.item()
            total_loss.append(global_loss)

            # 打印日志并记录到 TensorBoard
            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                # 计算剩余时间
                rest_time = spend_time / it * total_iters - spend_time
                rest_time = str(datetime.timedelta(seconds=rest_time))
                print(f"Epoch: {epoch + 1}/{args.epochs} | Step: {step + 1}/{iter_per_epoch} | Global Loss: {global_loss:.4f} | LR: {lr:.6f} | Seconds/Iteration: {spend_time / it:.4f} | Remaining time: {rest_time}")
                writer.add_scalar('Training Loss', global_loss, it)
                writer.add_scalar('Learning Rate', lr, it)
    
    # 绘制损失曲线
    plot_curve(total_loss, None, os.path.join(current_train_path, f"{model_name}_curve.png"))
    print(f"Curve saved to: {os.path.join(current_train_path, f'{model_name}_curve.png')}")
    model.save_pretrained(current_train_path)
    print(f"Model saved to: {current_train_path}")
    
    writer.close()  # 关闭 TensorBoard 日志记录器


# -------------------------------------------【主函数】------------------------------------------- #
def main():
    # 参数解析
    args = parse_args()

    if not torch.cuda.is_available():
         print(f"CUDA is not available. GPU is required for training.")
         return

    # 调用训练函数
    train_process(args)


if __name__ == '__main__':
    main()