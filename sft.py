import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
import time
import datetime
import math
from model import list_models, get_model_and_args
from utils.little_tools import create_folder, plot_loss_curve, save_to_yaml, load_yaml
from datasets import SFTDataset
import json
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("./mini_tokenizer")
vocab_size = len(tokenizer)
pad_id = tokenizer.unk_token_id  # TODO: 这里使用<unk>补全，后续可新增<pad>

# -------------------------------------------【参数解析】------------------------------------------- #
support_models = "、".join(list_models())
def parse_args():
    parser = argparse.ArgumentParser(description="SFT Mini Language Model")

    # 模型与训练精度
    parser.add_argument("--model_name", type=str, required=True, help=f"预训练模型名称，当前支持的模型有：{support_models}")
    parser.add_argument("--precision", type=str, default='bf16', help="选择使用混合精度训练: 默认 bf16, 可选 fp32 或 fp16")
    parser.add_argument("--model_ckpt_path", type=str, default="C:/Users/WKQ/Downloads/pretrained_model/mini_deepseekv3/pretrained_mini_deepseekv3_final_loss_3.08-base.pt", help="Base 模型权重路径")
    parser.add_argument("--model_args_path", type=str, default="C:/Users/WKQ/Downloads/pretrained_model/mini_deepseekv3/pretrained_mini_deepseekv3_model_args.yaml", help="模型配置路径")

    # ModelArgs 设置
    parser.add_argument("--max_batch_size", type=int, default=16, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=1, help="训练周期")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")  # 使用一个小的固定学习率微调
    
    # 路径与日志设置
    parser.add_argument("--sft_data_path", type=str, default='./preprocess_data/data/sft_data/sft_data_zh.csv', help="SFT 数据集路径")
    parser.add_argument("--output_path", type=str, default="./output", help="模型保存目录")
    parser.add_argument("--log_interval", type=int, default=10, help="训练日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10000, help="模型保存间隔")

    args = parser.parse_args()

    return args


# -------------------------------------------【训练函数】------------------------------------------- #
def train_process(args):

    # ------------------ 1. 设置 ------------------
    # 设置当前使用的设备
    device = torch.device("cuda")

    # 读取模型配置
    config = load_yaml(args.model_args_path)
    config['max_batch_size'] = args.max_batch_size
    config['epochs'] = args.epochs
    if args.model_name == 'mini_deepseekv3':  # 如果是 mini_deepseekv3 模型，禁用下面三项，只使用主 loss
        config['use_mtp'] = False  # 认为模型在预训练时已经有较好的预测能力，这里为了简化直接禁用
        config['use_noaux_tc'] = False  # 负载可能会受到<pad>的影响
        config['use_seq_aux'] = False  # 负载可能会受到<pad>的影响

    # ------------------ 2. 数据准备 ------------------
    dataset = SFTDataset(
        file_path=args.sft_data_path,
        tokenizer=tokenizer,
        max_seq_len=config['max_seq_len']
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.max_batch_size,
        pin_memory=True,  # 使用锁页内存提高数据加载速度
        num_workers=4,
        shuffle=True,
    )

    iter_per_epoch = len(dataloader)
    total_iters = args.epochs * iter_per_epoch  # 总迭代次数

    # --------------- 3. 模型与配置准备 --------------- 
    print(f"当前支持的模型有: {support_models}")
    print(f"当前加载的模型为: {args.model_name}")

    Model, Model_Args = get_model_and_args(args.model_name)  # 返回的是模型类和配置类

    # 实例化配置类，将命令行参数传入模型配置
    model_args = Model_Args(**config)
    model = Model(model_args).to(device)  # 实例化模型类，传入配置
    print(f"加载预训练模型权重: {args.model_ckpt_path}")
    state_dict = torch.load(args.model_ckpt_path, map_location=device)
    model.load_state_dict(state_dict) # 如果key完全匹配，直接加载

    # 配置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    
    print(f"模型参数量: {model.count_parameters()[1]}")
    print(f"模型配置: {json.dumps(asdict(model_args), indent=4)}")
    
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
        print("使用 FP16 混合精度训练")
    elif args.precision == 'bf16':
        autocast_dtype = torch.bfloat16
        enable_amp = True
        print("使用 BF16 混合精度训练")
    else:
        print("使用 FP32 精度训练")

    for epoch in range(model_args.epochs):
        model.train()  # 设置模型为训练模式
        for step, (X, Y) in enumerate(dataloader):
            
            X, Y = X.to(device), Y.to(device)

            # 清零梯度
            optimizer.zero_grad(set_to_none=True)

            it = epoch * iter_per_epoch + step + 1  # 当前全局迭代次数

            with autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):  # 使用混合精度训练
                _, loss, _ = model(X, Y)  # 由于SFTDataset设置了默认的ignore_index=-100，内部loss能够直接计算需生成部分的loss
            
            if scaler is not None: # 意味着正在使用 FP16
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # FP32 或 BF16
                loss.backward()
                optimizer.step()

            global_loss = loss.item()
            total_loss.append(global_loss)

            if it % args.save_interval == 0:
                # 保存模型时解开 DDP 包装
                model_path = os.path.join(current_train_path, f'{model_name}_epoch_{epoch+1}_iter_{it}_loss_{global_loss:.2f}-chat.pt')
                # 原子化保存（防写入中断）
                tmp_path = model_path + ".tmp"
                torch.save(model.state_dict(), tmp_path)
                os.rename(tmp_path, model_path)  # 重命名为最终文件名
                print(f"模型保存至: {model_path}")

            # 打印日志
            if step != 0 and step % args.log_interval == 0:
                spend_time = time.time() - start_time
                # 计算剩余时间
                rest_time = spend_time / it * total_iters - spend_time
                rest_time = str(datetime.timedelta(seconds=rest_time))
                print(f"Epoch: {epoch+1}/{model_args.epochs} | Step: {step+1}/{iter_per_epoch} | Global Loss: {global_loss:.4f} | 预计剩余训练时间: {rest_time}")
            writer.add_scalar('Training Loss', global_loss, it)
    
    # 绘制损失曲线
    plot_loss_curve(total_loss, os.path.join(current_train_path, f'{model_name}_loss_curve.png'))
    print(f"损失曲线保存至: {os.path.join(current_train_path, f'{model_name}_loss_curve.png')}")
    model_path = os.path.join(current_train_path, f'{model_name}_final_loss_{global_loss:.2f}-chat.pt')
    # 原子化保存（防写入中断）
    tmp_path = model_path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.rename(tmp_path, model_path)  # 重命名为最终文件名
    print(f"模型保存至: {model_path}")
    
    writer.close()  # 关闭 TensorBoard 日志记录器


# -------------------------------------------【主函数】------------------------------------------- #
def main():
    # 参数解析
    args = parse_args()

    if not torch.cuda.is_available():
         print(f"CUDA 不可用，需要使用 GPU 进行训练。")
         return

    # 调用训练函数
    train_process(args)


if __name__ == '__main__':
    main()



