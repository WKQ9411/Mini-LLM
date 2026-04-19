import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .utils import format_size, print_aligned


class DPODataset(Dataset):
    """
    DPO 偏好数据集, 直接从 jsonl 文件中读取 prompt / chosen / rejected

    每条样本会在初始化阶段完成 tokenize, 并使用和 SFT 相同的 masking 逻辑：
    prompt 部分 label 置为 ignore_index, 仅 assistant 回复部分参与 logprob 计算
    """

    def __init__(
        self,
        file_path: str,
        max_seq_len: int,
        tokenizer: PreTrainedTokenizerBase,
        min_score: int = 3,
        max_score: int = 5,
        ignore_index: int = -100,
    ):
        super().__init__()

        self.source_file_path = file_path
        self.max_seq_len = max_seq_len
        self.min_score = min_score
        self.max_score = max_score
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
        self.file_path = self._resolve_data_file_path(file_path)
        self.file_size_bytes = os.path.getsize(self.file_path)
        self.file_size = format_size(self.file_size_bytes)

        if self.tokenizer.pad_token_id is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            self.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer must provide either pad_token_id or eos_token_id.")

        self.data, self.stats = self._load_and_tokenize()

        info = {
            "file path": self.file_path,
            "file size": self.file_size,
            "max seq len": max_seq_len,
            "score range": f"[{min_score}, {max_score}]",
            "num samples": len(self.data),
            "loaded samples": self.stats["loaded"],
            "skipped score": self.stats["skipped_score"],
            "skipped invalid": self.stats["skipped_invalid"],
            "skipped too long": self.stats["skipped_too_long"],
        }
        if self.file_path != self.source_file_path:
            info["source file path"] = self.source_file_path
        print("-------------- dpo dataset info --------------")
        print_aligned(info)
        print("----------------------------------------------")

    def _resolve_data_file_path(self, file_path: str) -> str:
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"DPO data file not found: {file_path}")

        if self.min_score <= 0 and self.max_score >= 5:
            return str(source_path)

        candidate_source_path = source_path
        stem_match = re.match(r"^(?P<base>.+)_(?P<min>\d+)_(?P<max>\d+)$", source_path.stem)
        if stem_match:
            cached_min = int(stem_match.group("min"))
            cached_max = int(stem_match.group("max"))
            if cached_min == self.min_score and cached_max == self.max_score:
                print(f"Using score-range matched DPO file directly: {source_path}")
                return str(source_path)

            base_source_path = source_path.with_name(f"{stem_match.group('base')}{source_path.suffix}")
            if base_source_path.exists():
                candidate_source_path = base_source_path
            else:
                print(
                    f"Warning: base DPO dataset not found at {base_source_path}. "
                    f"Will build new cache from {source_path}."
                )

        cache_path = candidate_source_path.with_name(
            f"{candidate_source_path.stem}_{self.min_score}_{self.max_score}{candidate_source_path.suffix}"
        )
        if cache_path.exists():
            print(f"Using cached DPO subset: {cache_path}")
            return str(cache_path)

        self._build_score_cache_file(candidate_source_path, cache_path)
        return str(cache_path)

    def _build_score_cache_file(self, source_path: Path, cache_path: Path) -> None:
        kept = 0
        skipped_invalid = 0
        skipped_score = 0
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")

        print(
            f"Cached DPO subset not found. Building {cache_path.name} "
            f"from {source_path.name} for score range [{self.min_score}, {self.max_score}]..."
        )

        try:
            with open(source_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
                for line_no, line in enumerate(
                    tqdm(src, desc="Filtering DPO data"),
                    start=1,
                ):
                    raw = line.strip()
                    if not raw:
                        continue

                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        skipped_invalid += 1
                        print(f"Warning: invalid JSON in {source_path}:{line_no}, skipped while building cache.")
                        continue

                    normalized = self._normalize_record(item)
                    if normalized is None:
                        skipped_invalid += 1
                        continue

                    score = normalized["score"]
                    if score < self.min_score or score > self.max_score:
                        skipped_score += 1
                        continue

                    dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    kept += 1

            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        total = kept + skipped_score + skipped_invalid
        print(
            f"Saved cached DPO subset to: {cache_path} | "
            f"total: {total}, matched: {kept}, skipped score: {skipped_score}, skipped invalid: {skipped_invalid}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sft_collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """将 DPO 样本的 chosen 部分组装为 SFT 格式的 batch"""
        if not batch:
            raise ValueError("Batch is empty.")

        max_len = max(len(item["chosen_input_ids"]) for item in batch)

        input_ids, labels, attention_mask, position_ids = [], [], [], []
        for item in batch:
            features = self._pad_feature_group(
                input_ids=item["chosen_input_ids"],
                labels=item["chosen_labels"],
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

    def collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor | list[int]]:
        if not batch:
            raise ValueError("Batch is empty.")

        # 计算当前 batch 的最大序列长度，取最大值作为本批次的 padding 长度
        max_len = 0
        for item in batch:
            max_len = max(
                max_len,
                len(item["chosen_input_ids"]),
                len(item["rejected_input_ids"]),
            )

        chosen_input_ids = []
        chosen_labels = []
        chosen_attention_mask = []
        chosen_position_ids = []

        rejected_input_ids = []
        rejected_labels = []
        rejected_attention_mask = []
        rejected_position_ids = []

        sample_ids = []
        scores = []

        for item in batch:
            chosen_features = self._pad_feature_group(
                input_ids=item["chosen_input_ids"],
                labels=item["chosen_labels"],
                max_len=max_len,
            )
            rejected_features = self._pad_feature_group(
                input_ids=item["rejected_input_ids"],
                labels=item["rejected_labels"],
                max_len=max_len,
            )

            chosen_input_ids.append(chosen_features["input_ids"])
            chosen_labels.append(chosen_features["labels"])
            chosen_attention_mask.append(chosen_features["attention_mask"])
            chosen_position_ids.append(chosen_features["position_ids"])

            rejected_input_ids.append(rejected_features["input_ids"])
            rejected_labels.append(rejected_features["labels"])
            rejected_attention_mask.append(rejected_features["attention_mask"])
            rejected_position_ids.append(rejected_features["position_ids"])

            sample_ids.append(item["sample_id"])
            scores.append(item["score"])

        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_attention_mask, dtype=torch.long),
            "chosen_position_ids": torch.tensor(chosen_position_ids, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_attention_mask, dtype=torch.long),
            "rejected_position_ids": torch.tensor(rejected_position_ids, dtype=torch.long),
            "sample_ids": sample_ids,
            "scores": torch.tensor(scores, dtype=torch.long),
        }

    def _load_and_tokenize(self) -> tuple[list[dict], dict[str, int]]:
        data: list[dict] = []
        stats = {
            "loaded": 0,
            "skipped_score": 0,
            "skipped_invalid": 0,
            "skipped_too_long": 0,
        }

        print(f"Loading DPO data from jsonl file: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(tqdm(f, desc="Loading DPO data"), start=1):
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

                score = normalized["score"]
                if score < self.min_score or score > self.max_score:
                    stats["skipped_score"] += 1
                    continue

                chosen_features = self._build_preference_features(
                    prompt=normalized["prompt"],
                    response=normalized["chosen"],
                )
                rejected_features = self._build_preference_features(
                    prompt=normalized["prompt"],
                    response=normalized["rejected"],
                )

                if chosen_features is None or rejected_features is None:  # 大于 max_seq_len 的会被丢弃
                    stats["skipped_too_long"] += 1
                    continue

                data.append(
                    {
                        "sample_id": normalized["sample_id"],
                        "score": score,
                        "chosen_input_ids": chosen_features["input_ids"],
                        "chosen_labels": chosen_features["labels"],
                        "rejected_input_ids": rejected_features["input_ids"],
                        "rejected_labels": rejected_features["labels"],
                    }
                )
                stats["loaded"] += 1

        return data, stats

    def _normalize_record(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        prompt = self._safe_text(item.get("prompt"))
        chosen = self._safe_text(item.get("chosen"))
        rejected = self._safe_text(item.get("rejected"))
        if not prompt or not chosen or not rejected:
            return None

        try:
            score = int(item.get("score"))
        except (TypeError, ValueError):
            return None
        if score < 0 or score > 5:
            return None

        sample_id = item.get("sample_id")
        if sample_id is None:
            return None

        return {
            "sample_id": sample_id,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "score": score,
        }

    def _build_preference_features(
        self,
        prompt: str,
        response: str,
    ) -> dict[str, list[int]] | None:
        user_message = {"role": "user", "content": prompt}
        assistant_message = {"role": "assistant", "content": response}

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
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        if prompt_len >= len(full_ids):
            return None
        if len(full_ids) > self.max_seq_len:
            return None

        labels = [self.ignore_index] * prompt_len + full_ids[prompt_len:]
        if len(labels) != len(full_ids):
            return None

        return {
            "input_ids": full_ids,
            "labels": labels,
        }

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

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()
