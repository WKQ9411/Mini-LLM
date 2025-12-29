from huggingface_hub import HfApi, login
from pathlib import Path
import os
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def upload_model_to_hf(
    model_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    include_tokenizer: bool = True,
    tokenizer_path: str = None
):
    """
    上传模型到 Hugging Face Hub
    
    Args:
        model_path: 本地模型路径 (包含 config.json 和 model.safetensors)
        repo_id: Hugging Face 仓库 ID, 格式为 "username/model_name"
        token: Hugging Face token, 如果为 None 则从环境变量 HF_TOKEN 获取
        private: 是否创建私有仓库
        include_tokenizer: 是否同时上传 tokenizer
        tokenizer_path: tokenizer 路径，如果为 None 则使用默认路径
    """
    # 获取 token
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("Please provide Hugging Face token, can be set via parameter --token or environment variable HF_TOKEN")
    
    # 登录
    login(token=token)
    
    # 初始化 API
    api = HfApi()
    
    # 创建仓库（如果不存在）
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"Repository {repo_id} already exists or has been created")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # 转换为 Path 对象
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # 上传模型文件
    print(f"Uploading model files from {model_path} to {repo_id}...")
    
    # 需要上传的模型文件
    model_files = ["config.json", "model.safetensors", "generation_config.json"]
    
    for file_name in model_files:
        file_path = model_path / file_name
        if file_path.exists():
            print(f"Uploading {file_name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {file_name}"
            )
            print(f"✓ {file_name} uploaded successfully")
        else:
            print(f"⚠ File does not exist, skipping: {file_name}")
    
    # 上传 tokenizer（如果需要）
    if include_tokenizer:
        root_path = Path(__file__).parent.parent
        if tokenizer_path is None:
            tokenizer_path = root_path / "mini_tokenizer"
        else:
            tokenizer_path = Path(tokenizer_path)
        
        if tokenizer_path.exists():
            print(f"Uploading tokenizer from {tokenizer_path}...")
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
            
            for file_name in tokenizer_files:
                file_path = tokenizer_path / file_name
                if file_path.exists():
                    print(f"Uploading {file_name}...")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_name,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Upload tokenizer {file_name}"
                    )
                    print(f"✓ {file_name} uploaded successfully")
                else:
                    print(f"⚠ File does not exist, skipping: {file_name}")
        else:
            print(f"⚠ Tokenizer path does not exist: {tokenizer_path}")
    
    print(f"\n✓ Model uploaded successfully! Access: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload model weights to Hugging Face Hub")

    parser.add_argument("--model_path", type=str, required=True, help="Local model path (e.g. output/pretrained_mini_llama3)")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID (e.g. username/model_name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (can also be set via environment variable HF_TOKEN)")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--no-tokenizer", action="store_true", help="Don't upload tokenizer")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (default: mini_tokenizer)")
    
    args = parser.parse_args()
    
    upload_model_to_hf(
        model_path=args.model_path,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        include_tokenizer=not args.no_tokenizer,
        tokenizer_path=args.tokenizer_path
    )


if __name__ == "__main__":
    # Example:
    # python .\scripts\upload_weights.py --model_path D:\Code\MyProject\Mini-LLM\output\pretrained_mini_llama3 --repo_id WKQ9411/Mini-Llama3-100M-Base
    main()