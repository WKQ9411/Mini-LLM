# Mini-LLM

<p align="center">
    <img src="./assets/logo.png" width="300"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/WKQ9411">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/datasets/wangkunqing/mini_llm_dataset">ModelScope</a>&nbsp&nbsp
<br>
<a href="README.md">Chinese</a>&nbsp&nbsp
</p>

# Changelog

- [2026-02-01] Implemented `mini_qwen3_next` model; optimized multi-turn conversation data construction; optimized `mini_models` structure.
- [2025-12-29] Project initialization; implemented `mini_llama3` and `mini_deepseekv3` models; implemented pretrain and sft.

# I. Project Introduction

This project aims to replicate mainstream open-source model architectures with limited computational resources, implementing mini models with 100-200M parameters. The project fixes datasets, training pipelines, and other infrastructure as much as possible, so that when learning new model architectures, they can be quickly reproduced, allowing the main focus to be on learning and reproducing model architectures.

**Main Goals:**
1. Learn and reproduce mainstream open-source model architectures
2. Implement common training and inference pipelines from scratch

To achieve this goal, in previous versions of Mini-LLM, we fully customized the `model` package, including base classes such as `BaseModel` and `BaseModelArgs`. Later, we discovered that this approach is similar to transformers library's `PreTrainedModel` and `PretrainedConfig`. Based on this similarity, to better integrate with the HuggingFace ecosystem, we directly refactored the project structure. The current version's models are fully compatible with the transformers library and can directly use methods like `from_pretrained`, `generate` for model loading and inference. At the same time, to deeply understand the principles of training and inference, the project still provides a set of independent training code and generation code implementations. The early version of Mini-LLM has been moved to the legacy branch.

# II. How to Reproduce This Project from Scratch Step by Step

## (I) Environment Setup

1. Clone the project

```shell
git clone https://github.com/WKQ9411/Mini-LLM.git
cd Mini-LLM
```

2. Initialize environment

Execute the following script to automatically detect the environment and install dependencies. Model training requires a CUDA environment. If you only want to try inference with pre-trained weights, a CPU environment is sufficient.

```shell
# Linux
bash ./scripts/setup.sh

# Windows
.\scripts\setup.ps1
```

## (II) Dataset Preparation

### 1. Dataset Introduction

Currently used datasets include:

1. [OpenCSG Fineweb-Edu-Chinese-V2.1 Dataset](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1)

Mainly using the high-quality corpus portion with scores of 4-5 from this dataset, totaling 9745 parquet files (numbered from 0-9766, with some missing numbers in between, but 9745 files are indeed downloaded from modelscope), approximately 70GB in total. Main uses:
- Use 5% sampling of this dataset as tokenizer training data
- Use 20% sampling of this dataset as pre-training data
- If computational resources are sufficient, the full dataset can also be used for pre-training

2. [DeepCtrl Large Model Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)

Approximately 16GB, main uses:
- Use the entire dataset as pre-training data
- Sample 50,000 entries as SFT data

### 2. Dataset Preparation

Dataset-related scripts are located in the `scripts` folder. You can download the required datasets using the following commands:

```shell
# Linux
bash ./scripts/download_data.sh

# Windows
.\scripts\download_data.ps1
```

The interface is shown in the figure below, where you can select the dataset number to download:

<div align="center">
<img src="./assets/download_data_script.png" width="90%" alt="download_data">
</div>

Among them:

- [1] Download a .parquet format data subset sampled at 5% from the OpenCSG Fineweb-Edu-Chinese-V2.1 dataset for training tokenizer (you can also directly use the pre-trained tokenizer, located in the project's `mini_tokenizer` folder)
- [2] Download **all original** .parquet files with scores 4-5 from the OpenCSG Fineweb-Edu-Chinese-V2.1 dataset for pre-training
- [3] Download a .parquet format data subset sampled at 20% from the OpenCSG Fineweb-Edu-Chinese-V2.1 dataset for pre-training. Sampling is done proportionally by category, maintaining the same distribution as the original dataset:
<div align="center">
<img src="./assets/sampled_source_distribution.png" width="60%" alt="sampled_source_distribution">
</div>

- [4] Download a .bin format data subset sampled at 20% from the OpenCSG Fineweb-Edu-Chinese-V2.1 dataset for pre-training (processed into token ids by `mini_tokenizer`)
- [5] Download all .bin format data files from the DeepCtrl large model dataset for pre-training (processed into token ids by `mini_tokenizer`)
- [6] Download **all original** .jsonl format data files from the DeepCtrl large model dataset for SFT
- [7] Download processed .parquet format data files from the DeepCtrl large model dataset for SFT (processed into token ids by `mini_tokenizer`, including: (a) all eligible SFT data converted to parquet format; (b) sampled 50,000 entries and 200 self-awareness entries; (c) data after packing (b); it is recommended to use (c) for SFT). The length distribution of sampled data is as follows:

<div align="center">
<img src="./assets/sampled_sft_data_length_distribution.png" width="70%" alt="sampled_sft_data_length_distribution">
</div>

You can choose to directly download processed data for training (recommended), or download raw data and process it yourself. Data processing code is located at:

```shell
./scripts/prepare_tokenizer_data.py
./scripts/prepare_pretrain_data.py
./scripts/prepare_sft_data.py
```

This project currently uses: [1] for training tokenizer, [4]+[5] for pre-training (merge .bin files through the `merge_pretrain_data` function in `prepare_pretrain_data.py`), and c. from [7] for SFT.

## (III) Training Tokenizer

The new version of mini_tokenizer is consistent with Qwen, using special tokens including: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`.
The base vocabulary size is 32,000 (including `<|endoftext|>`), and `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>` are added as added tokens to the vocabulary, so the vocabulary size is 32,004 (among them, `<think>` and `</think>` are currently unused, and will be considered for use when updating RL-related code in the future). The chat template is located at `data/tokenizer_data/chat_template.jinja2`. The current chat template is only based on `user`, `assistant`, and automatically matches the content of the `think` part.
You can directly use the pre-trained `mini_tokenizer`, or retrain it. To retrain, execute:

```shell
python ./train/train_tokenizer.py
```

If you need to retrain the tokenizer, it is recommended to ensure the CPU has sufficient RAM. If 5% sampled data is still too large for the tokenizer being trained, you can use a smaller `sample_ratio` in `scripts/prepare_tokenizer_data.py` to sample a smaller tokenizer dataset.

## (IV) Model Architecture

Model architecture references papers, official repository source code, transformers implementations, etc. The `hidden_states` shape is unified as: `(B, H, L, D)`, where `B` is batch size, `H` is the number of heads, `L` is sequence length, and `D` is the dimension per head.

For model architecture, please refer to my [GitHub Blog](https://wkq9411.github.io/):

> The code sections for `mini_llama3` and `mini_deepseekv3` in the blog are based on earlier versions of Mini-LLM. While they are not fully consistent with the current version, the core concepts are the same.

1. `mini_llama3`, Dense Model:
   - [Code Analysis](https://wkq9411.github.io/2026-01-01/Code-Llama3.html)
2. `mini_deepseekv3`, MoE Model:
   - [Paper Analysis](https://wkq9411.github.io/2026-01-01/Paper-DeepSeek-V3.html)
   - [Code Analysis](https://wkq9411.github.io/2026-01-01/Code-DeepSeek-V3.html)
3. `mini_qwen3_next`, Linear Model:
   - [Paper Analysis - Transformers are RNNs](https://wkq9411.github.io/2026-01-18/Paper-Transformers-are-RNNs.html)
   - [Paper Analysis - Gated Delta Network](https://wkq9411.github.io/2026-01-18/Paper-Gated-Delta-Network.html)
   - [Paper Analysis - Gated Attention](https://wkq9411.github.io/2026-01-18/Paper-Gated-Attention.html)

## (V) Pre-training

Training with a single GPU:

```shell
python ./train/pretrain.py --model_name=mini_deepseekv3 --max_batch_size=32
```

Training with DDP:

```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ./train/pretrain.py --model_name=mini_deepseekv3 --max_batch_size=32
```

For more training parameter descriptions, please refer to the `parse_args()` function in `train/pretrain.py`.

After training starts, open tensorboard in a new terminal to monitor training progress:

```shell
tensorboard --logdir=output/
```

If using a cloud server, configure tensorboard port and other parameters and set public access according to different platform documentation, so that training progress can be monitored locally, for example:

```bash
tensorboard --logdir=output/ --port=8080 --bind_all
```

Training records common metrics such as `learning_rate`, `loss`, `ppl`, etc. In addition, taking the `mini_deepseekv3` model as an example, it also records additional metrics including **expert load balancing**, **sequence-level auxiliary loss**, **mtp loss**, etc., as shown in the following figures:

<div align="center">
<img src="./assets/load_balance.png" width="90%" alt="load_balance">
</div>

<div align="center">
<img src="./assets/training_progress.png" width="90%" alt="training_progress">
</div>

> Among them, the expert load curve records the ratio of maximum/minimum activation counts of all experts in each layer. A value approaching 1 indicates load balancing, and larger values indicate load imbalance.

## (VI) SFT

Since the model parameters are basically 100-200M and SFT training data is relatively small, single GPU training is sufficient:

```shell
python ./train/sft.py --model_name=mini_deepseekv3 --max_batch_size=32
```

For more training parameter descriptions, please refer to the `parse_args()` function in `train/sft.py`.

SFT dataset can choose whether to use packing dataset. After enabling packing, computational resources can be effectively utilized, and the actual effective token length of each batch can be as consistent as possible, thereby avoiding gradient dilution issues. After using packing, each batch needs to construct the corresponding `attention_mask`, visualized as follows (packed two entries):

<div align="center">
<img src="./assets/attention_mask_visualization.png" width="60%" alt="attention_mask_visualization">
</div>

After packing, the SFT curve is relatively smoother.
- Packing curve:
<div align="center">
<img src="./assets/packing_sft.png" width="60%" alt="packing_sft">
</div>

- Unpacked curve:
<div align="center">
<img src="./assets/no_packing_sft.png" width="60%" alt="no_packing_sft">
</div>

## (VII) Inference

Inference demo code is located in the `example` folder. You can use the project's custom `Generator` class for inference, or use transformers' native `generate` method for inference.

Run in terminal:

```shell
python ./example/test_terminal.py --model_name=mini_deepseekv3
```

For more inference parameter descriptions, please refer to the `parse_args()` function in `example/test_terminal.py`.

You can also perform inference via API, providing it to popular frontends for dialogue (wrapped with `wrap-openai` to provide OpenAI-compatible API, you can refer to my other repository [wrap-openai](https://github.com/WKQ9411/wrap-openai)). Start the backend with the following command:

```shell
python ./example/test_api.py --model_name=mini_deepseekv3
```

Taking [CherryStudio](https://www.cherry-ai.com/) as an example, after configuring the OpenAI-compatible API, the dialogue effect is as follows:

<div align="center">
<img src="./assets/example.gif" width="100%" alt="example">
</div>

In addition, the model parameters of this project have been uploaded to HuggingFace and can be directly downloaded and used. Usage methods can be found in `example/use_example.ipynb`.

> Due to the small model parameter size, while it may predict the next token relatively well to some extent, this does not mean it has good generalization ability, knowledge base, or reasoning ability. Small models are more likely to "remember" surface patterns in training data (such as specific phrases, sentence structures, formats) rather than truly "understand" their meaning. This causes them to easily produce hallucinations and incoherent outputs when facing prompts that require knowledge, reasoning, or slightly deviate from training patterns.

