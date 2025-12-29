# 此脚本用于将数据集上传到 modelscope

from modelscope.hub.api import HubApi
from dotenv import load_dotenv
import os
from pathlib import Path


load_dotenv()
root_path = Path(__file__).parent.parent
data_path = root_path / "data"


ACCESS_TOKEN = os.getenv("MODELSCOPE_ACCESS_TOKEN")
api = HubApi()
api.login(ACCESS_TOKEN)


owner_name = 'wangkunqing'
dataset_name = 'mini_llm_dataset'


api.upload_file(
    path_or_fileobj=str(data_path / "sft_data/sft_parquet.zip"),  # 本地的文件名
    path_in_repo='./sft_parquet.zip',  # repo 的文件名
    repo_id=f"{owner_name}/{dataset_name}",
    repo_type = 'dataset',
    commit_message='upload dataset file to repo',
)