from prepare_pretrain_data import analyze_and_sample_fineweb_edu
from pathlib import Path


# 定义路径
root_path = Path(__file__).parent.parent
# pretrain
pretrain_data_path = root_path / "data/pretrain_data"
# tokenizer
tokenizer_data_path = root_path / "data/tokenizer_data"
# OpenCSG Fineweb-Edu-Chinese-V2.1 数据集 - pretrain
# https://www.modelscope.cn/datasets/opencsg/Fineweb-Edu-Chinese-V2.1
fineweb_edu_file_path = pretrain_data_path / "fineweb_edu"


if __name__ == "__main__":
    
    # tokenizer 训练数据通过从 OpenCSG Fineweb-Edu-Chinese-V2.1 数据集中抽样 5% 得到
    print("=" * 30)
    print("Start processing tokenizer datasets...")
    print("=" * 30)

    # --------------------------------- 处理 tokenizer 数据集 ---------------------------------
    sample_ratio = 0.05
    fineweb_edu_sampled_output_path = str(tokenizer_data_path / f"fineweb_edu_sampled_{int(sample_ratio*100)}_percent")
    analyze_and_sample_fineweb_edu(
        data_path=str(fineweb_edu_file_path), 
        output_path=str(fineweb_edu_sampled_output_path), 
        sample_ratio=sample_ratio
        )