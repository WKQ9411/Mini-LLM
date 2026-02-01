from wrap_openai import register_funcs, run_server
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
from pathlib import Path
from mini_models import get_model_and_config, list_models, get_model_info, Generator
import torch
import json
import argparse


root_path = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, help=f"Model name, support: {', '.join(list_models())}")
    parser.add_argument("--suffix", type=str, default="", help="Suffix, e.g. '_1', '_2', ... or empty")
    parser.add_argument("--weight_path", type=str, default=None, help="Weight path, if not provided, will use the default output directory")

    parser.add_argument("--generate_func", type=str, default="custom", choices=["custom", "transformers"], help="Generate function: 'custom' or 'transformers'")
    parser.add_argument("--port", type=int, default=9411, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--require-api-key", type=lambda x: x.lower() in ["true", "1", "yes"], default=True, help="Require API Key")

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


def generate(messages, model, tokenizer, max_new_tokens, temperature, top_p, top_k, repetition_penalty, frequency_penalty, args):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    if args.generate_func == "transformers":
        inputs.pop("token_type_ids", None)

        # 创建流式生成器
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # generate 参数
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            use_cache=True,
        )

        # 如果模型是 mini_qwen3_next，手动初始化 MiniQwen3NextDynamicCache 以避免 transformers 默认初始化 DynamicCache 导致冲突
        if args.model_name == "mini_qwen3_next":
            from mini_models.cache import MiniQwen3NextDynamicCache
            generation_kwargs["past_key_values"] = MiniQwen3NextDynamicCache(model.config)
        
        # 在单独线程中运行生成
        generation_thread = Thread(target=model.generate, kwargs=generation_kwargs)
        generation_thread.start()
        
        # 从流式生成器中逐个yield文本
        for text_chunk in streamer:
            yield text_chunk
    
    elif args.generate_func == "custom":
        generator = Generator(model, tokenizer)
        for text_chunk in generator.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty
        ):
            yield text_chunk
    else:
        raise ValueError(f"Invalid generate function: {args.generate_func}")


def main():
    # 解析参数
    args = parse_args()

    # 设置权重路径
    if args.weight_path is None:
        args.weight_path = str(root_path / f"output/sft_{args.model_name}{args.suffix}")

    print(f"Loading {args.model_name} from {args.weight_path}")
    tokenizer, model = load_model(args)
    print(f"{args.model_name} loaded successfully!")
    print(f"Using device: {model.device}")
    print(f"Model info: {json.dumps(get_model_info(model)[1], indent=2)}")
    print(f"Generate function: {args.generate_func}")

    # 调用 wrap-openai 封装 openai 兼容 api
    register_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": args.frequency_penalty,
        "args": args,
    }

    register_funcs(generate_func=generate, support_stream=True, **register_kwargs)

    print(f"\nServer is running at http://{args.host}:{args.port}")
    print(f"API endpoints: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"Health check: http://{args.host}:{args.port}/health")
    if args.require_api_key:
        print(f"API Key is required")
        print(f"Use the following command to generate API Key:")
        print(f"    Generate API Key: wrap-openai --generate --name \"my_key\"")
        print(f"    List API Keys   : wrap-openai --list")

    run_server(host=args.host, port=args.port, require_api_key=args.require_api_key)

if __name__ == "__main__":
    main()