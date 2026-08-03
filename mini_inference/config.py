import os
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.7
    tensor_parallel_size: int = 1
    dtype: torch.dtype | None = None
    enforce_eager: bool = False
    attention_backend: Literal["triton", "flash_attn"] = "triton"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.attention_backend not in {"triton", "flash_attn"}:
            raise ValueError(
                "attention_backend must be either 'triton' or 'flash_attn'"
            )
        if self.hf_config is None:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        if self.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("mini_inference dtype must be torch.float16 or torch.bfloat16")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
