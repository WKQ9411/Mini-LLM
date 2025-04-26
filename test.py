from model import get_model_and_args
import torch
from transformers import AutoTokenizer
from utils.little_tools import load_yaml


tokenizer_name = "./mini_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# 模型权重路径
root_path = "C:/Users/WKQ/Downloads/pretrained_model/"
model_name = "mini_deepseekv3"
model_path = root_path + model_name + "/pretrained_mini_deepseekv3_epoch_1_iter_200000_loss_3.31-base.pt"
yaml_path = root_path + model_name + "/pretrained_mini_deepseekv3_model_args.yaml"
# model_name = "mini_llama3"
# model_path = "C:\\Users\\WKQ\\Downloads\\pretrained_mini_llama3_epoch_1_iter_150000_loss_2.937476396560669-base.pt"

config = load_yaml(yaml_path)
config['max_batch_size'] = 1

# 初始化模型
Model, Model_Args = get_model_and_args(model_name)
model_args = Model_Args(**config)
model = Model(model_args)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu" 

model.load_state_dict(torch.load(model_path, map_location=device))

_, approx_params = model.count_parameters()
print(f'模型参数量：{approx_params}')
print(f'最大输入长度：{config["max_seq_len"]}')

# 对话循环
while True:
    prompt = input("input:\n")
    if prompt == "q":
        break
    print('output:')
    output_generator = model.generate(prompt=prompt, context_length=512, tokenizer=tokenizer, stream=True, task="generate", temperature=1.1)
    for output in output_generator:
        print(output, end='', flush=True)
    print('\n')