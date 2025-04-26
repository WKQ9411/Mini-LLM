from model import list_models, get_model_and_args
from transformers import AutoTokenizer
import argparse


tokenizer_name = "./mini_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

support_models = "、".join(list_models())
parser = argparse.ArgumentParser(description="Pretrain Mini Language Model")

# 模型与训练精度
parser.add_argument("--model_name", type=str, required=True, help=f"预训练模型名称，当前支持的模型有：{support_models}")
args = parser.parse_args()

if args.model_name not in list_models():
    raise ValueError(f"不支持的模型名称：{args.model_name}，当前支持的模型有：{support_models}")

Model, Model_Args = get_model_and_args(args.model_name)

model_args = Model_Args(vocab_size=len(tokenizer))
model = Model(model_args)

print(f"{args.model_name} 的参数量：{model.count_parameters()[1]}")
