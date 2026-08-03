import argparse
import json
from pathlib import Path
from threading import Thread

import torch
from transformers import AutoTokenizer, TextIteratorStreamer

from mini_models import Generator, get_model_and_config, get_model_info, list_models


root_path = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Mini-LLM Chat Test")

    parser.add_argument("--model_name", type=str, required=True, help=f"Model name, support: {', '.join(list_models())}")
    parser.add_argument("--suffix", type=str, default="", help="Suffix, e.g. '_1', '_2', ... or empty")
    parser.add_argument("--weight_path", type=str, default=None, help="Weight path, if not provided, will use the default output directory")
    parser.add_argument("--tokenizer_path", type=str, default=str(root_path / "mini_tokenizer"), help="Tokenizer directory")

    parser.add_argument(
        "--generate_func",
        type=str,
        default="custom",
        choices=["custom", "transformers", "mini_inference"],
        help="Generation backend: custom, transformers, or mini_inference",
    )
    parser.add_argument("--chat_mode", type=str, default="chat", choices=["chat", "generation"], help="Chat mode: 'chat' for chat model or 'generation' for pretrained model")
    parser.add_argument("--max_history_messages", type=int, default=5, help="Max history messages, only used in chat mode")
    parser.add_argument("--enable_think", action="store_true", help="Enable think-mode prompt prefix in chat mode")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory fraction used by mini_inference")
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA Graph for mini_inference")
    
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--frequency_penalty", type=float, default=0.3, help="Frequency penalty")

    args = parser.parse_args()
    if args.max_history_messages <= 0:
        parser.error("--max_history_messages must be greater than 0")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be greater than 0")
    if args.temperature < 0:
        parser.error("--temperature must be greater than or equal to 0")
    if args.top_k < 0:
        parser.error("--top_k must be greater than or equal to 0")
    if not 0.0 <= args.top_p <= 1.0:
        parser.error("--top_p must be in [0, 1]")
    if args.repetition_penalty < 1.0:
        parser.error("--repetition_penalty must be greater than or equal to 1")
    if args.frequency_penalty < 0.0:
        parser.error("--frequency_penalty must be greater than or equal to 0")
    if (
        args.generate_func == "mini_inference"
        and args.repetition_penalty != 1.0
        and args.frequency_penalty != 0.0
    ):
        parser.error("--repetition_penalty and --frequency_penalty cannot be enabled together")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu_memory_utilization must be in (0, 1]")
    return args


def _resolve_device_and_dtype():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    return torch.device("cpu"), None


def load_hf_model(args):
    """加载 mini_models 模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    Model, Config = get_model_and_config(args.model_name)
    device, dtype = _resolve_device_and_dtype()
    load_kwargs = {"dtype": dtype} if dtype is not None else {}

    config = Config.from_pretrained(args.weight_path)

    model = Model.from_pretrained(args.weight_path, config=config, **load_kwargs)
    model = model.to(device)
    model.eval()  # 设置为评估模式

    return tokenizer, model


def load_mini_inference(args):
    """加载独立的 Mini-Inference Engine"""
    if args.model_name != "mini_llama3":
        raise ValueError("mini_inference currently only supports model_name=mini_llama3")
    if not torch.cuda.is_available():
        raise RuntimeError("mini_inference requires a CUDA device")

    from mini_inference import LLM

    llm = LLM(
        args.weight_path,
        tokenizer_path=args.tokenizer_path,
        tensor_parallel_size=1,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    return llm.tokenizer, llm


def _use_think_mode(args) -> bool:
    return args.chat_mode == "chat" and args.enable_think


def _apply_chat_template(tokenizer, messages, args, tokenize: bool = False) -> str | list[int]:
    template_kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": True,
    }
    if tokenize:
        template_kwargs["return_dict"] = False
    if _use_think_mode(args):
        template_kwargs["enable_think"] = True
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def _emit_prefilled_think(args) -> str:
    if not _use_think_mode(args):
        return ""
    think_prefix = "<think>\n"
    print(think_prefix, end="", flush=True)
    return think_prefix


def generate_with_custom(messages, model, tokenizer, args):
    """使用自定义 Generator 生成回复，返回完整文本"""
    generator = Generator(model, tokenizer)
    
    # 构建输入
    if args.chat_mode == "chat":
        # 使用聊天模板
        formatted_text = _apply_chat_template(tokenizer, messages, args)
        input_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"].to(model.device)
    else:
        # generation mode，确保只使用最后一条消息
        input_text = messages[-1]["content"] if messages else ""
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(model.device)
    
    # 流式生成并收集完整文本
    full_response = _emit_prefilled_think(args)
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
        formatted_text = _apply_chat_template(tokenizer, messages, args)
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

    # 对使用自定义 cache 协议的模型，手动初始化 past_key_values，避免 transformers 默认 DynamicCache 导致冲突
    if args.model_name == "mini_qwen3_next":
        from mini_models.cache import MiniQwen3NextDynamicCache
        generation_kwargs["past_key_values"] = MiniQwen3NextDynamicCache(model.config)
    elif args.model_name == "mini_deepseekv4":
        from transformers.cache_utils import Cache
        from mini_models.cache import MiniDeepSeekV4CacheLayer
        generation_kwargs["past_key_values"] = Cache(
            layers=[MiniDeepSeekV4CacheLayer(model.config) for _ in range(model.config.num_hidden_layers)]
        )
    
    # 移除 None 值
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    
    # 使用 TextIteratorStreamer 来收集输出
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs["streamer"] = streamer
    
    # 在单独线程中运行生成
    generation_thread = Thread(target=model.generate, kwargs=generation_kwargs)
    generation_thread.start()
    
    # 流式输出并收集完整文本
    full_response = _emit_prefilled_think(args)
    for text in streamer:
        print(text, end="", flush=True)
        full_response += text
    
    generation_thread.join()
    return full_response


def generate_with_mini_inference(messages, llm, tokenizer, args):
    """使用 Mini-Inference Engine 流式生成回复"""
    if args.chat_mode == "chat":
        prompt_token_ids = _apply_chat_template(tokenizer, messages, args, tokenize=True)
    else:
        input_text = messages[-1]["content"] if messages else ""
        prompt_token_ids = tokenizer.encode(input_text)

    max_model_len = llm.model_runner.config.max_model_len
    available_tokens = max_model_len - len(prompt_token_ids)
    if available_tokens <= 0:
        raise ValueError(
            f"Prompt length {len(prompt_token_ids)} reaches the model limit {max_model_len}; "
            "use 'clear' to reset conversation history."
        )

    from mini_inference import SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        max_tokens=min(args.max_new_tokens, available_tokens),
    )
    full_response = _emit_prefilled_think(args)
    for text_chunk in llm.stream_generate(prompt_token_ids, sampling_params):
        print(text_chunk, end="", flush=True)
        full_response += text_chunk
    return full_response


def generate_response(messages, model, tokenizer, args):
    """根据 generate_func 选择生成后端并返回完整文本"""
    if args.generate_func == "custom":
        return generate_with_custom(messages, model, tokenizer, args)
    if args.generate_func == "transformers":
        return generate_with_transformers(messages, model, tokenizer, args)
    if args.generate_func == "mini_inference":
        return generate_with_mini_inference(messages, model, tokenizer, args)
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
    if args.generate_func == "mini_inference":
        tokenizer, model = load_mini_inference(args)
    else:
        tokenizer, model = load_hf_model(args)
    print(f"{args.model_name} loaded successfully!")
    if args.generate_func == "mini_inference":
        config = model.model_runner.config
        print("Using device: cuda")
        print(f"Dtype: {config.dtype}")
        print(f"Max model length: {config.max_model_len}")
        print(f"CUDA Graph: {not config.enforce_eager}")
    else:
        print(f"Using device: {model.device}")
        print(f"Model info: {json.dumps(get_model_info(model)[1], indent=2)}")
    print(f"Generate function: {args.generate_func}")
    print(f"Chat mode: {args.chat_mode}")
    print(f"Enable think: {args.enable_think}")
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
