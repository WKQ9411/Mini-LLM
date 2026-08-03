from wrap_openai import register_generate, run_server
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Lock, Thread
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

    parser.add_argument(
        "--generate_func",
        type=str,
        default="custom",
        choices=["custom", "transformers", "mini_inference"],
        help="Generation backend: custom, transformers, or mini_inference",
    )
    parser.add_argument("--enable_think", action="store_true", help="Enable think-mode prompt prefix for chat completions")
    parser.add_argument("--port", type=int, default=9411, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--require-api-key", type=lambda x: x.lower() in ["true", "1", "yes"], default=True, help="Require API Key")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory fraction used by mini_inference")
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA Graph for mini_inference")

    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--frequency_penalty", type=float, default=0.3, help="Frequency penalty")

    args = parser.parse_args()
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
    tokenizer = AutoTokenizer.from_pretrained(str(root_path / "mini_tokenizer"))

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
        tokenizer_path=str(root_path / "mini_tokenizer"),
        tensor_parallel_size=1,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    return llm.tokenizer, llm


def _apply_chat_template(tokenizer, messages, args) -> str:
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.enable_think:
        template_kwargs["enable_think"] = True
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def generate(
    messages,
    model,
    tokenizer,
    max_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    frequency_penalty,
    args,
    generation_lock,
):
    if args.generate_func == "mini_inference":
        from mini_inference import SamplingParams

        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": False,
        }
        if args.enable_think:
            template_kwargs["enable_think"] = True
        prompt_token_ids = tokenizer.apply_chat_template(messages, **template_kwargs)

        max_model_len = model.model_runner.config.max_model_len
        available_tokens = max_model_len - len(prompt_token_ids)
        if available_tokens <= 0:
            raise ValueError(
                f"Prompt length {len(prompt_token_ids)} reaches the model limit {max_model_len}"
            )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=min(max_tokens, available_tokens),
        )
        with generation_lock:
            if args.enable_think:
                yield "<think>\n"
            yield from model.stream_generate(prompt_token_ids, sampling_params)
        return

    prompt = _apply_chat_template(tokenizer, messages, args)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if args.enable_think:
        yield "<think>\n"
    
    if args.generate_func == "transformers":
        inputs.pop("token_type_ids", None)

        # 创建流式生成器
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # generate 参数
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
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
            max_new_tokens=max_tokens,
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
    if args.generate_func == "mini_inference":
        tokenizer, model = load_mini_inference(args)
        generation_lock = Lock()
    else:
        tokenizer, model = load_hf_model(args)
        generation_lock = None
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
    print(f"Enable think: {args.enable_think}")

    # 调用 wrap-openai 封装 openai 兼容 api
    register_generate(
        generate_func=generate,
        support_stream=True,
        model_id=args.model_name,
        fixed_kwargs={
            "model": model,
            "tokenizer": tokenizer,
            "args": args,
            "generation_lock": generation_lock,
        },
        openai_kwargs={
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "frequency_penalty": args.frequency_penalty,
        },
        custom_kwargs={
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
    )

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
