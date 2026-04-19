import json
import os
import random
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .utils import format_size, print_aligned


class GRPODataset(Dataset):
    """
    GRPO 数据集, 从 jsonl 文件中读取 prompt / thinking / response

    每条样本会在初始化阶段完成 tokenize, 同时构建两种格式:
    1. SFT 格式: 完整的 prompt + <think>thinking</think> + response, 用于冷启动 SFT
    2. Prompt 格式: 仅 prompt 部分, 用于 GRPO 训练时模型自行生成
    """

    def __init__(
        self,
        file_path: str,
        max_seq_len: int,
        tokenizer: PreTrainedTokenizerBase,
        num_samples: int | None = None,
        seed: int = 42,
        ignore_index: int = -100,
    ):
        super().__init__()

        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.num_samples = num_samples  # 可选，用于从数据集中随机采样一部分子集
        self.seed = seed
        self.ignore_index = ignore_index

        self.file_size_bytes = os.path.getsize(file_path)
        self.file_size = format_size(self.file_size_bytes)

        if self.tokenizer.pad_token_id is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            self.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer must provide either pad_token_id or eos_token_id.")

        self.data, self.stats = self._load_and_tokenize()

        # 随机子集采样
        if num_samples is not None and num_samples < len(self.data):
            self.data = random.Random(seed).sample(self.data, num_samples)

        info = {
            "file path": self.file_path,
            "file size": self.file_size,
            "max seq len": max_seq_len,
            "num samples": len(self.data),
            "loaded samples": self.stats["loaded"],
            "skipped invalid": self.stats["skipped_invalid"],
            "skipped too long": self.stats["skipped_too_long"],
        }
        if num_samples is not None:
            info["sampled subset"] = num_samples
        print("------------- grpo dataset info --------------")
        print_aligned(info)
        print("----------------------------------------------")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ----------------------------------------- 数据加载 -----------------------------------------

    def _load_and_tokenize(self) -> tuple[list[dict], dict[str, int]]:
        data: list[dict] = []
        stats = {
            "loaded": 0,
            "skipped_invalid": 0,
            "skipped_too_long": 0,
        }

        print(f"Loading GRPO data from jsonl file: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(tqdm(f, desc="Loading GRPO data"), start=1):
                raw = line.strip()
                if not raw:
                    continue

                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    stats["skipped_invalid"] += 1
                    print(f"Warning: invalid JSON in {self.file_path}:{line_no}, skipped.")
                    continue

                normalized = self._normalize_record(item)
                if normalized is None:
                    stats["skipped_invalid"] += 1
                    continue

                sft_features = self._build_sft_features(
                    prompt=normalized["prompt"],
                    thinking=normalized["thinking"],
                    response=normalized["response"],
                )
                prompt_features = self._build_prompt_features(
                    prompt=normalized["prompt"],
                )

                if sft_features is None or prompt_features is None:
                    stats["skipped_too_long"] += 1
                    continue

                data.append({
                    "sample_id": normalized["sample_id"],
                    "sft_input_ids": sft_features["input_ids"],
                    "sft_labels": sft_features["labels"],
                    "prompt_input_ids": prompt_features["input_ids"],
                    "prompt_text": normalized["prompt"],
                    "ground_truth_response": normalized["response"],
                })
                stats["loaded"] += 1

        return data, stats

    def _normalize_record(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        prompt = self._safe_text(item.get("prompt"))
        thinking = self._safe_text(item.get("thinking"))
        response = self._safe_text(item.get("response"))
        if not prompt or not thinking or not response:
            return None

        sample_id = item.get("sample_id")
        if sample_id is None:
            return None

        return {
            "sample_id": sample_id,
            "prompt": prompt,
            "thinking": thinking,
            "response": response,
        }

    # ----------------------------------------- 特征构建 -----------------------------------------

    def _build_sft_features(
        self,
        prompt: str,
        thinking: str,
        response: str,
    ) -> dict[str, list[int]] | None:
        """构建 SFT 格式特征: prompt 部分和 label 置为 ignore_index"""
        user_message = {"role": "user", "content": prompt}
        # 将 thinking 用 <think> 标签包裹, 与 response 一起作为 assistant 回复
        assistant_content = f"<think>\n{thinking}\n</think>\n\n{response}"
        assistant_message = {"role": "assistant", "content": assistant_content}

        full_text = self.tokenizer.apply_chat_template(
            [user_message, assistant_message],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        prompt_text = self.tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            enable_think=True,
        )  # 加入 <think> 标签，模型要做的是生成 think 内容并正确闭合 </think>，模型无需主动生成第一个 <think> 标签
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        if prompt_len >= len(full_ids):
            return None
        if len(full_ids) > self.max_seq_len:
            return None

        labels = [self.ignore_index] * prompt_len + full_ids[prompt_len:]
        return {"input_ids": full_ids, "labels": labels}

    def _build_prompt_features(self, prompt: str) -> dict[str, list[int]] | None:
        """构建 prompt-only 特征, 用于 GRPO 训练时模型自行生成"""
        user_message = {"role": "user", "content": prompt}
        prompt_text = self.tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            enable_think=True,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > self.max_seq_len:
            return None
        return {"input_ids": prompt_ids}

    # ----------------------------------------- Padding -----------------------------------------

    def _pad_feature_group(
        self,
        input_ids: list[int],
        labels: list[int],
        max_len: int,
    ) -> dict[str, list[int]]:
        pad_len = max_len - len(input_ids)
        if pad_len < 0:
            raise ValueError("pad_len must be non-negative.")

        return {
            "input_ids": input_ids + [self.pad_token_id] * pad_len,
            "labels": labels + [self.ignore_index] * pad_len,
            "attention_mask": [1] * len(input_ids) + [0] * pad_len,
            "position_ids": list(range(max_len)),
        }

    # ---------------------------------------- Collate Fn ----------------------------------------

    def sft_collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """组装为 SFT 格式的 batch, 用于冷启动微调"""
        if not batch:
            raise ValueError("Batch is empty.")

        max_len = max(len(item["sft_input_ids"]) for item in batch)

        input_ids, labels, attention_mask, position_ids = [], [], [], []
        for item in batch:
            features = self._pad_feature_group(
                input_ids=item["sft_input_ids"],
                labels=item["sft_labels"],
                max_len=max_len,
            )
            input_ids.append(features["input_ids"])
            labels.append(features["labels"])
            attention_mask.append(features["attention_mask"])
            position_ids.append(features["position_ids"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }

    def prompt_collate_fn(self, batch: list[dict]) -> dict[str, Any]:
        """组装为 prompt-only 格式的 batch, 用于 GRPO 训练时生成补全"""
        if not batch:
            raise ValueError("Batch is empty.")

        max_len = max(len(item["prompt_input_ids"]) for item in batch)

        input_ids = []
        attention_mask = []
        prompt_texts = []
        sample_ids = []
        prompt_lengths = []
        ground_truth_responses = []

        for item in batch:
            ids = item["prompt_input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append([self.pad_token_id] * pad_len + ids)  # 左 pad
            attention_mask.append([0] * pad_len + [1] * len(ids))
            prompt_texts.append(item["prompt_text"])
            sample_ids.append(item["sample_id"])
            prompt_lengths.append(len(ids))
            ground_truth_responses.append(item["ground_truth_response"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_texts": prompt_texts,
            "sample_ids": sample_ids,
            "prompt_lengths": prompt_lengths,
            "ground_truth_responses": ground_truth_responses,
        }

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()
