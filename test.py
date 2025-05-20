from model import get_model_and_args
import torch
from transformers import AutoTokenizer
from utils.little_tools import load_yaml


tokenizer_name = "./mini_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# 模型权重路径
model_name = "mini_deepseekv3"
model_path = "C:/Users/WKQ/Downloads/sft_model/mini_deepseekv3/sft_mini_deepseekv3_final_loss_2.09-chat.pt"
yaml_path = "C:/Users/WKQ/Downloads/sft_model/mini_deepseekv3/sft_mini_deepseekv3_model_args.yaml"
# model_name = "mini_llama3"
# model_path = "C:/Users/WKQ/Downloads/sft_model/mini_llama3/sft_mini_llama3_final_loss_2.52-chat.pt"
# yaml_path = "C:/Users/WKQ/Downloads/sft_model/mini_llama3/sft_mini_llama3_model_args.yaml"


config = load_yaml(yaml_path)
config['max_batch_size'] = 1
if model_name == "mini_deepseekv3":
    config['use_noaux_tc'] = True  # 确保推理时无辅助损失负载均衡策略的bias起作用

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
    output_generator = model.generate(
        prompt=prompt, 
        context_length=model_args.max_seq_len, 
        tokenizer=tokenizer, 
        stream=True, 
        task="chat",  # 根据任务类型修改
        temperature=0.8, 
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.0,
        frequency_penalty=0.5
        )
    for output in output_generator:
        print(output, end='', flush=True)
    print('\n')