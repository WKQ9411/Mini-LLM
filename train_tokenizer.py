import json
import os
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)


data_path = './preprocess_data/data/pretrain_data/'
data_name = 'wikipedia-cn-20230720-filtered.json'  # https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/tree/main
data_file = data_path + data_name

# 初始化tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 定义特殊token
special_tokens = ["<unk>", "<s>", "</s>"]

# 设置训练器并添加特殊token
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=special_tokens,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 使用ByteLevel的默认字母表
)

def read_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as file:
        data = json.load(file)  # data 是 Python 列表
        for item in data:
            yield item['completion']

# 读取文本数据
texts = read_data(data_file)
# 训练tokenizer
tokenizer.train_from_iterator(texts, trainer=trainer)

# 设置解码器
tokenizer.decoder = decoders.ByteLevel()

# 保存tokenizer
tokenizer_dir = "./mini_tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save(tokenizer_dir)

# 手动创建配置文件
config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": True,
    "added_tokens_decoder": {
        "0": {
            "content": "<unk>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "1": {
            "content": "<s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "2": {
            "content": "</s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            }
    },
    "bos_token": "<s>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "</s>",
    "legacy": True,
    "model_max_length": 1000000,
    "pad_token": None,
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<unk>",
    "use_default_system_prompt": False,
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
}

# 保存配置文件
with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)

print("Tokenizer 保存成功！")