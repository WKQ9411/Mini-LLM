from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
from pathlib import Path
from mini_models import get_model_and_config, list_models, get_model_info, Generator
import torch
import json
import argparse


root_path = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Mini-LLM Chat Test")

    parser.add_argument("--model_name", type=str, required=True, help=f"Model name, support: {', '.join(list_models())}")
    parser.add_argument("--suffix", type=str, default="", help="Suffix, e.g. '_1', '_2', ... or empty")
    parser.add_argument("--weight_path", type=str, default=None, help="Weight path, if not provided, will use the default output directory")

    parser.add_argument("--generate_func", type=str, default="custom", choices=["custom", "transformers"], help="Generate function: 'custom' or 'transformers'")
    parser.add_argument("--chat_mode", type=str, default="chat", choices=["chat", "generation"], help="Chat mode: 'chat' for chat model or 'generation' for pretrained model")
    parser.add_argument("--max_history_messages", type=int, default=5, help="Max history messages, only used in chat mode")
    
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--frequency_penalty", type=float, default=0.3, help="Frequency penalty")

    return parser.parse_args()


def load_model(args):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))

    Model, Config = get_model_and_config(args.model_name)
    model = Model.from_pretrained(args.weight_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # 设置为评估模式

    return tokenizer, model


def generate_with_custom(messages, model, tokenizer, args):
    """使用自定义 Generator 生成回复，返回完整文本"""
    generator = Generator(model, tokenizer)
    
    # 构建输入
    if args.chat_mode == "chat":
        # 使用聊天模板
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"].to(model.device)
    else:
        # generation mode，确保只使用最后一条消息
        input_text = messages[-1]["content"] if messages else ""
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(model.device)
    
    # 流式生成并收集完整文本
    full_response = ""
    for text_chunk in generator.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
    ):
        print(text_chunk, end="", flush=True)
        full_response += text_chunk
    
    return full_response


def generate_with_transformers(messages, model, tokenizer, args):
    """使用 transformers 原生的 generate 方法生成回复，返回完整文本"""
    
    # 构建输入
    if args.chat_mode == "chat":
        # 使用聊天模板
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)
    else:
        # generation mode，确保只使用最后一条消息
        input_text = messages[-1]["content"] if messages else ""
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    inputs.pop("token_type_ids", None)
    
    # 生成参数
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        top_p=args.top_p if args.top_p > 0 and args.top_p < 1.0 else None,
        top_k=args.top_k if args.top_k > 0 else None,
        temperature=args.temperature if args.temperature > 0 else None,
        repetition_penalty=args.repetition_penalty if args.repetition_penalty != 1.0 else None,
        use_cache=True,
    )

    # 如果模型是 mini_qwen3_next，手动初始化 MiniQwen3NextDynamicCache 以避免 transformers 默认初始化 DynamicCache 导致冲突
    if args.model_name == "mini_qwen3_next":
        from mini_models.cache import MiniQwen3NextDynamicCache
        generation_kwargs["past_key_values"] = MiniQwen3NextDynamicCache(model.config)
    
    # 移除 None 值
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    
    # 使用 TextIteratorStreamer 来收集输出
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer
    
    # 在单独线程中运行生成
    generation_thread = Thread(target=model.generate, kwargs=generation_kwargs)
    generation_thread.start()
    
    # 流式输出并收集完整文本
    full_response = ""
    for text in streamer:
        print(text, end="", flush=True)
        full_response += text
    
    generation_thread.join()
    return full_response


def generate_response(messages, model, tokenizer, args):
    """生成回复, 根据 generate_func 选择使用自定义 Generator 或 transformers 原生的 generate 方法，返回完整文本"""
    if args.generate_func == "custom":
        return generate_with_custom(messages, model, tokenizer, args)
    elif args.generate_func == "transformers":
        return generate_with_transformers(messages, model, tokenizer, args)
    else:
        raise ValueError(f"Invalid generate function: {args.generate_func}")


def main():
    # 解析参数
    args = parse_args()

    # 设置权重路径
    prefix = "sft" if args.chat_mode == "chat" else "pretrained"
    if args.weight_path is None:
        args.weight_path = str(root_path / f"output/{prefix}_{args.model_name}{args.suffix}")

    print("======== Mini-LLM Chat Test ========")
    print(f"Loading {args.model_name} from {args.weight_path}")
    tokenizer, model = load_model(args)
    print(f"{args.model_name} loaded successfully!")
    print(f"Using device: {model.device}")
    print(f"Model info: {json.dumps(get_model_info(model)[1], indent=2)}")
    print(f"Generate function: {args.generate_func}")
    print(f"Chat mode: {args.chat_mode}")
    if args.chat_mode == "chat":
        print(f"Max history messages: {args.max_history_messages}")

    print("\n--------------------------------")
    print("Type 'quit' or 'exit' to exit the chat.")
    if args.chat_mode == "chat":
        print("Type 'clear' to clear conversation history.")
    print("--------------------------------")

    # 历史消息管理，仅 chat mode 使用
    history_messages = []

    while True:
        try:
            # 获取用户输入
            input_text = input("\nUser: ").strip()
            
            # 检查退出命令
            if input_text.lower() in ['quit', 'exit']:
                print("Bye!")
                break
            
            # 检查 clear 命令
            if args.chat_mode == "chat" and input_text.lower() == 'clear':
                history_messages = []
                print("Conversation history cleared.")
                continue
            
            # 空输入处理
            if not input_text:
                continue

            # 构建消息列表
            if args.chat_mode == "chat":
                # chat mode: 维护历史消息
                history_messages.append({"role": "user", "content": input_text})
                messages = history_messages.copy()
            else:
                # generation mode: 只使用当前输入，不保留历史
                messages = [{"role": "user", "content": input_text}]

            # 生成并显示回复
            print("Mini-LLM: ", end="", flush=True)
            full_response = generate_response(messages, model, tokenizer, args)  # 流式输出
            print()  # 输出换行

            # 将助手回复添加到历史，并限制历史消息数量
            if args.chat_mode == "chat":
                history_messages.append({"role": "assistant", "content": full_response.strip()})
                # 限制历史消息数量
                if len(history_messages) > args.max_history_messages * 2:
                    # 保留最近的对话
                    history_messages = history_messages[-args.max_history_messages * 2:]

        except KeyboardInterrupt:
            print("\n\nBye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
