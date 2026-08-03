from copy import copy
from enum import Enum, auto
from itertools import count

from mini_inference.sampling_params import SamplingParams


# 枚举类型，用于表示序列的状态
class SequenceStatus(Enum):
    WAITING = auto()   # 等待 prefill 的序列
    RUNNING = auto()   # 已经 prefill 正在 decode 的序列
    FINISHED = auto()  # 生成完毕的序列


class Sequence:
    block_size = 256  # pagedattention 中一个物理 block 所能容纳的 token 数，即分页的页大小
    counter = count()  # 给每个 Sequence 实例自动分配一个全局唯一、自增的 id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 序列状态
        self.seq_id = next(Sequence.counter)  # 实例 id
        self.status = SequenceStatus.WAITING  # 初始化序列状态
        self.is_prefill = True  # 是否为 prefill
        self.block_table = []  # 存储 block 表
        
        # token 相关
        self.token_ids = copy(token_ids)  # 存储 token_ids，使用 copy 以避免外部修改影响内部状态
        self.last_token = token_ids[-1]  # 最后一个 token
        self.num_tokens = len(self.token_ids)  # 总 token 数量
        self.num_prompt_tokens = len(token_ids)  # prompt token 数量，初始化时等于总 token 数量
        self.num_cached_tokens = 0  # 缓存的 token 数量，初始化为 0
        self.num_scheduled_tokens = 0  # 已调度的 token 数量，初始化为 0
        self.completion_token_counts = {}  # 只统计生成 token，供重复惩罚和频率惩罚使用
        
        # 采样参数
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.repetition_penalty = sampling_params.repetition_penalty
        self.frequency_penalty = sampling_params.frequency_penalty
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.completion_token_counts[token_id] = self.completion_token_counts.get(token_id, 0) + 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
