# Mini-LLM

<p align="center">
   <img src="./assets/logo.png" width="300"/>
</p>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/WKQ9411">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-WKQ9411-ffc107?color=ffc107&logoColor=white"/>
  </a>
  <a href="https://www.modelscope.cn/datasets/wangkunqing/mini_llm_dataset">
    <img alt="ModelScope" src="https://img.shields.io/badge/%F0%9F%A4%96%20ModelScope-mini__llm__dataset-624aff?color=624aff&logoColor=white"/>
  </a>
  <a href="./assets/wechat.png">
    <img alt="WeChat" src="https://img.shields.io/badge/WeChat-QR%20Code-brightgreen?logo=wechat&logoColor=white"/>
  </a>
  <br>
  <a href="README_en.md">
    <img alt="English" src="https://img.shields.io/badge/Docs-English-blue"/>
  </a>
</div>

# 更新日志

- [2026-04-19] 增加 YaRN、DPO、GRPO
- [2026-02-01] 实现 `mini_qwen3_next` 模型；优化多轮对话数据构造；优化 `mini_models` 结构。
- [2025-12-29] 完成兼容 transformers 的项目重构；实现 `mini_llama3`、`mini_deepseekv3` 模型；实现 pretrain、SFT。

# 一、项目简介

本项目旨在基于较小的算力，复现当前主流开源模型的架构，实现一个 100-200M 参数量版本的迷你模型。项目将数据集、训练流程等基础设施尽可能固定下来，以便在学习新的模型架构时能够快速复现，从而将主要精力聚焦在模型架构的学习和复现上。

**主要目标：**
1. 学习并复现当前主流开源模型架构
2. 从零实现常用的训练和推理流程

为了实现这一目标，在先前版本的 Mini-LLM 中，我们完全自定义实现了 `model` 包，其中包括 `BaseModel` 和 `BaseModelArgs` 等基类。后来发现，这样的构建思路与 transformers 库的 `PreTrainedModel` 和 `PretrainedConfig` 类似。基于这种相似性，为了更好地与 HuggingFace 生态兼容，我们直接重构了项目结构。当前版本实现的模型完全兼容 transformers 库，可以直接使用 `from_pretrained`、`generate` 等方法进行模型加载和推理。同时，为了深入理解训练和推理原理，项目仍然提供了一套独立的训练代码和生成代码实现。早期版本的 Mini-LLM 已移动到 legacy 分支。

# 二、如何从零一步一步复现本项目

## （一）环境配置

1. 克隆项目

```shell
git clone https://github.com/WKQ9411/Mini-LLM.git
cd Mini-LLM
```

2. 初始化环境

执行以下脚本，将自动检测环境并安装依赖。模型训练需要 CUDA 环境，如果仅尝试使用训练好的权重进行推理，可以使用 CPU 环境。

```shell
# Linux
bash ./scripts/setup.sh

# Windows
.\scripts\setup.ps1
```

## （二）数据集准备

### 1. 数据集介绍

目前使用的数据集包括：

1. [OpenCSG Fineweb-Edu-Chinese-V2.1 数据集](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1)

主要使用该数据集中得分为4-5分的高质量语料部分，共9745个 parquet 文件（编号是从0-9766，中间似乎有缺失编号，但从 ModelScope 上确实下载的是9745个文件），总共约70GB。主要用途：
- 使用该数据集的5%采样作为 tokenizer 训练数据
- 使用该数据集的20%采样作为预训练数据
- 如果算力足够，也可采用全部数据进行预训练
- 使用该数据集的0.1%采样作为YaRN微调数据

2. [匠数科技大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)

大约16GB，主要用途：
- 使用该数据集的全部作为预训练数据
- 采样50000条作为 SFT 数据

3. [OpenCSG UltraFeedback-chinese 数据集](https://huggingface.co/datasets/opencsg/UltraFeedback-chinese)

使用其中的 `ultrafeedback-chinese-binarized-lowest` 子集作为 DPO 数据集，可根据 DPO 评分范围筛选训练子集。

4. 自行合成的 GRPO 数据集。

数据集由 Gemini 3.1 Pro 合成，任务是修复或修改 JSON 格式，分为 prompt、thinking、response 三个部分。prompt 用于给出一个有错误或需修改的简单 JSON，thinking 是合成的思维链，response 是不包含额外解释的 JSON 输出。

### 2. 数据集准备

数据集相关脚本位于`scripts`文件夹中，可以通过以下命令下载需要的数据集：

```shell
# Linux
bash ./scripts/download_data.sh

# Windows
.\scripts\download_data.ps1
```

界面如下图所示，可选择需要下载的数据集序号进行下载：

<div align="center">
<img src="./assets/download_data_script.png" width="90%" alt="download_data">
</div>


其中：

- [1] 【Tokenizer】下载按照5%的比例对OpenCSG Fineweb-Edu-Chinese-V2.1数据集进行采样的.parquet格式数据子集，用于训练tokenizer（你也可以直接使用训练好的tokenizer，文件位于项目的`mini_tokenizer`文件夹内）
- [2] 【Pretrain】下载**原始**OpenCSG Fineweb-Edu-Chinese-V2.1数据集中得分4-5的全部.parquet文件，用于预训练
- [3] 【Pretrain】下载按照20%的比例对OpenCSG Fineweb-Edu-Chinese-V2.1数据集进行采样的.parquet格式数据子集，用于预训练，采样按照类别等比例采样，分布与原始数据集一致：
<div align="center">
<img src="./assets/sampled_source_distribution.png" width="60%" alt="sampled_source_distribution">
</div>

- [4] 【Pretrain】下载按照20%的比例对OpenCSG Fineweb-Edu-Chinese-V2.1数据集进行采样的.bin格式数据子集，用于预训练（已通过`mini_tokenizer`处理为token ids）
- [5] 【Pretrain】下载OpenCSG Fineweb-Edu-Chinese-V2.1数据集中得分4-5的全部.bin格式数据文件，用于预训练（已通过`mini_tokenizer`处理为token ids）
- [6] 【Pretrain】下载匠数科技大模型数据集的全部.bin格式数据文件，用于预训练（已通过`mini_tokenizer`处理为token ids）
- [7] 【YaRN】下载按照0.1%的比例对OpenCSG Fineweb-Edu-Chinese-V2.1数据集进行采样的.bin格式数据子集，用于YaRN微调（已通过`mini_tokenizer`处理为token ids）
- [8] 【SFT】下载**原始**匠数科技大模型数据集的全部.jsonl格式数据文件，用于SFT
- [9] 【SFT】下载经过处理的匠数科技大模型数据集的.parquet格式数据文件，用于SFT（已通过`mini_tokenizer`处理为token ids，其中包括：(a) 转化为parquet格式的全部符合条件的SFT数据；(b) 采样的50000条采样数据和200条自我认知数据；(c) 将 (b) 进行packing处理后的数据；推荐使用 (c) 进行SFT），采样后的数据长度分布如下：

<div align="center">
<img src="./assets/sampled_sft_data_length_distribution.png" width="70%" alt="sampled_sft_data_length_distribution">
</div>

- [10] 【DPO】下载经过处理的DPO数据集
- [11] 【GRPO】下载合成的GRPO数据集

你可以选择直接下载处理好的数据进行训练（推荐），也可以选择下载原始数据自行处理，处理数据的代码位于：

```shell
./scripts/prepare_tokenizer_data.py
./scripts/prepare_pretrain_data.py
./scripts/prepare_sft_data.py
./scripts/prepare_dpo_data.py
./scripts/prepare_grpo_data.py
```

本项目当前使用：[1] 训练 tokenizer，[4]+[6] 进行预训练（通过`prepare_pretrain_data.py`中的`merge_pretrain_data`函数合并.bin文件），[9]中的(c)进行 SFT。

## （三）训练 tokenizer

新版本的 mini_tokenizer 与 Qwen 保持一致，使用的特殊 token 包括：`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`。
基础词表大小为32000（包括`<|endoftext|>`），`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`作为 added tokens 添加到词表中，因此词表大小为32004。tokenizer 使用方法可以参考`example/tokenizer_example.ipynb`。聊天模板位于`data/tokenizer_data/chat_template.jinja2`。

可直接使用训练好的`mini_tokenizer`，或重新训练。重新训练执行：

```shell
python ./train/train_tokenizer.py
```

如果需要重新训练 tokenizer，建议确保 CPU 具有较大的 RAM。如果5%采样数据对于所训练的 tokenizer 仍然较大，可以在`scripts/prepare_tokenizer_data.py`中使用更小的`sample_ratio`采样更小一点的 tokenizer 数据集。

## （四）模型结构

模型结构参考论文、官方仓库源码、transformers 实现等，其中的`hidden_states`形状统一为: `(B, H, L, D)`，其中`B`为 batch size，`H`为头数，`L`为序列长度，`D`为每个头的维度。

模型结构参考本人 [GitHub 博客](https://wkq9411.github.io/)：

> 博客中的 `mini_llama3` 和 `mini_deepseekv3` 的代码部分是基于早期版本的 Mini-LLM 实现的，与当前版本不完全一致，但核心思想是相同的。

1. `mini_llama3`，Dense Model:
   - [代码解读](https://wkq9411.github.io/2026-01-01/Code-Llama3.html)
2. `mini_deepseekv3`，MoE Model:
   - [论文解读](https://wkq9411.github.io/2026-01-01/Paper-DeepSeek-V3.html)
   - [代码解读](https://wkq9411.github.io/2026-01-01/Code-DeepSeek-V3.html)
3. `mini_qwen3_next`，Linear Model:
   - [论文解读 - Transformers are RNNs](https://wkq9411.github.io/2026-01-18/Paper-Transformers-are-RNNs.html)
   - [论文解读 - Gated Delta Network](https://wkq9411.github.io/2026-01-18/Paper-Gated-Delta-Network.html)
   - [论文解读 - Gated Attention](https://wkq9411.github.io/2026-01-18/Paper-Gated-Attention.html)

## （五）预训练

使用单卡进行训练：

```shell
python ./train/pretrain.py --model_name=mini_deepseekv3 --max_batch_size=32
```

使用 DDP 进行训练：

```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ./train/pretrain.py --model_name=mini_deepseekv3 --max_batch_size=32
```

更多训练参数说明，请参考`train/pretrain.py`中的`parse_args()`函数。

训练开始后，在新的终端内开启 TensorBoard 用于监控训练进度：

```shell
tensorboard --logdir=output/
```

如果使用的是云服务器，根据不同平台使用文档，配置 TensorBoard 端口等参数并设置公开访问等，从而能够在本地监控训练进度，例如：

```bash
tensorboard --logdir=output/ --port=8080 --bind_all
```

训练会记录通用的`learning_rate`、`loss`、`ppl`等指标，此外，以`mini_deepseekv3`模型为例，还额外记录了包括**专家负载情况**、**序列级辅助损失**、**MTP 损失**等指标，如下图所示：

<div align="center">
<img src="./assets/load_balance.png" width="90%" alt="load_balance">
</div>

<div align="center">
<img src="./assets/training_progress.png" width="90%" alt="training_progress">
</div>

> 其中，专家负载曲线通过记录每层的所有专家最大/最小激活次数的比值，趋近于1表示负载均衡，越大表示负载不均衡。

## （六）SFT

由于模型参数量基本在100-200M，SFT 训练数据又相对较少，仅使用单卡训练即可：

```shell
python ./train/sft.py --model_name=mini_deepseekv3 --max_batch_size=32
```

更多训练参数说明，请参考`train/sft.py`中的`parse_args()`函数。

SFT 数据集可选择是否使用 packing 数据集，开启 packing 后，能够有效利用算力，同时让每个 batch 的实际有效 token 长度尽可能一致，从而避免梯度稀释问题。使用 packing 后，每个 batch 需要构造相应的`attention_mask`，可视化如下（pack了两条数据）：

<div align="center">
<img src="./assets/attention_mask_visualization.png" width="60%" alt="attention_mask_visualization">
</div>

packing 后，SFT 曲线相对更加平滑。
- packing 的曲线：
<div align="center">
<img src="./assets/packing_sft.png" width="60%" alt="packing_sft">
</div>

- 未 packing 的曲线：
<div align="center">
<img src="./assets/no_packing_sft.png" width="60%" alt="no_packing_sft">
</div>

> 由于 Linear Model 的特点，当前暂不支持 mini_qwen3_next 的 packing SFT，见[Issues #3](https://github.com/WKQ9411/Mini-LLM/issues/3)

## （七）YaRN

关于 YaRN 的理论部分，可以参考博客：[YaRN 论文解读](https://wkq9411.github.io/2026-01-01/Paper-YaRN.html)

在模型配置中传入 `rope_scaling` 参数，例如：

```python
rope_scaling = {
   "rope_type": "yarn",
   "factor": 4.0,
   "attention_factor": None,  # 默认为 None，内部自动计算
   "beta_fast": 32,
   "beta_slow": 1,
}
```

可选对少量长文本进行微调：

```shell
python ./train/yarn.py --model_name mini_llama3 --max_seq_len 2048
```

更多训练参数说明，请参考 `train/yarn.py` 中的 `parse_args()` 函数。

对加入YaRN之前、加入YaRN但不微调、加入YaRN并微调三种情况计算长文本的PPL：

```shell
python ./eval/eval_yarn.py --base_model_path output/pretrained_mini_llama3 --yarn_finetuned_model_path output/yarn_mini_llama3
```

结果如下：

<div align="center">
<img src="./assets/eval_yarn_ppl_curve.png" width="90%" alt="yarn">
</div>

可见加入YaRN后，相比原始模型，长文本的PPL有明显下降。此外，加入YaRN并微调后，长文本的PPL进一步下降，但在微调长度范围之外，PPL会逐渐上升，这可能是因为微调后模型学习到了新的位置编码语义，其外推能力反而弱于仅插入 YaRN 而不微调的模型。因此，若目标是固定长度范围内的长文本建模，可以选择对少量长文本继续微调；若更关注超过训练长度后的外推表现，则可以只在推理或评估时加入 YaRN，而不进行额外微调。

## （八）DPO

关于 DPO 的理论部分，可以参考博客：[DPO 论文解读](https://wkq9411.github.io/2026-03-11/Paper-DPO.html)

运行如下代码：

```shell
python ./train/dpo.py --model_name mini_llama3
```

更多训练参数说明，请参考 `train/dpo.py` 中的 `parse_args()` 函数。由于小模型DPO很容易训崩，因此通常使用较小的学习率和参数 $\beta$，此外，还提供了可选的 DPOP、冻结参数、预先对 chosen 进行 SFT 等优化措施。

其中，DPOP 是 DPO 的一个变体，在 DPO Loss 后加入了一个针对 chosen response 的正样本约束项：当 policy model 对 chosen response 的 log probability 低于 reference model 时，额外加入 `lambda * max(ref_chosen_logp - policy_chosen_logp, 0)` 作为惩罚，从而避免模型为了拉大 chosen 和 rejected 的偏好差距而降低 chosen 本身的概率。在本项目的小模型设置下，DPOP 通常比直接 DPO 更稳定，可以通过 `--loss_type dpop --dpop_lambda 1.0` 开启。

DPO 训练结果如下：

<div align="center">
<img src="./assets/dpop+sft_warmup_more_aggressive.png" width="90%" alt="dpo_progress">
</div>

具体结果对比请参考 `eval/eval_dpo.ipynb`。

## （九）GRPO

关于 GRPO 的理论部分，可以参考博客：[从策略梯度到PPO再到GRPO](https://wkq9411.github.io/2026-03-22/RL-PG-PPO-GRPO.html)

本项目中的 GRPO 示例任务是 JSON 修复或修改：模型根据 prompt 中给出的错误 JSON 或修改要求，先在 `<think>...</think>` 中给出简短思考过程，再在 JSON 代码块中输出最终 JSON。冷启动时，为了使模型仍然保留一定的指令遵循能力，避免模型输出迅速坍缩到 JSON 任务的狭窄模式里，加入了一部分通用数据。最终，冷启动数据包含800条通用数据和1200条 JSON 任务数据。

当前奖励函数主要由四部分组成：输出格式奖励、思考长度奖励、JSON 可解析奖励和正确性奖励。其中正确性奖励权重最高，用于鼓励模型输出与标准答案一致的 JSON；其余奖励用于约束输出结构，避免小模型在 RL 阶段产生不可解析或格式混乱的结果。

运行如下代码：

```shell
python ./train/grpo.py --model_name mini_llama3 --max_batch_size 4 --cold_start_sft --sft_batch_size 16 --sft_epochs 3 --grpo_epochs 2
```

更多训练参数说明，请参考 `train/grpo.py` 中的 `parse_args()` 函数。默认情况下，GRPO 会从 `output/sft_{model_name}` 加载 SFT 后的模型作为初始 policy；开启 `--cold_start_sft` 后，会先使用 `data/grpo_data/cold_start.jsonl` 做一轮任务格式冷启动 SFT，再进入 GRPO 训练。若希望在 JSON 任务语料上继续预训练，也可以额外开启 `--mid_training`（不过可能影响不大）。

训练结果如下：

<div align="center">
<img src="./assets/grpo_curve.png" width="90%" alt="grpo_curve">
</div>

从结果看，GRPO 后模型在格式遵循和 JSON 可解析性上进一步提升，同时正确率相比仅进行 cold-start SFT 有明显提高。具体结果对比请参考 `eval/eval_grpo.ipynb`。

## （十）推理

相关推理 demo 的代码位于`example`文件夹中，可以使用项目自定义的`Generator`类进行推理，也可以使用 transformers 原生的`generate`方法进行推理。

在终端中运行：

```shell
python ./example/test_terminal.py --model_name=mini_deepseekv3
```

更多推理参数说明，请参考`example/test_terminal.py`中的`parse_args()`函数。

也可以通过 API 方式进行推理，提供给流行的前端进行对话（通过`wrap-openai`封装 OpenAI 兼容 API，可参考我的另一个仓库[wrap-openai](https://github.com/WKQ9411/wrap-openai)），通过如下命令启动后端：

```shell
python ./example/test_api.py --model_name=mini_deepseekv3
```

以[CherryStudio](https://www.cherry-ai.com/)为例，配置好OpenAI兼容API后，对话效果如下：

<div align="center">
<img src="./assets/example.gif" width="100%" alt="example">
</div>

此外，本项目的模型参数已上传至HuggingFace，可直接下载使用，调用方法见`example/use_example.ipynb`。

> 由于模型参数量较小，虽然可能一定程度上较好的预测下一个token，但是并不等同于它具备了良好的泛化能力、知识储备或推理能力。小模型更容易“记住”训练数据中的表面模式（比如特定短语、句子结构、格式），而不是真正“理解”其含义。这导致它们在面对需要知识、推理或稍微偏离训练模式的prompt时，容易产生幻觉和不连贯的输出。

# Star History

<div align="center">
  <a href="https://www.star-history.com/?repos=WKQ9411%2FMini-LLM&type=date&logscale=&legend=top-left">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=WKQ9411/Mini-LLM&type=date&theme=dark&legend=top-left" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=WKQ9411/Mini-LLM&type=date&legend=top-left" />
      <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=WKQ9411/Mini-LLM&type=date&legend=top-left" />
    </picture>
  </a>
</div>
