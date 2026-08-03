import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from mini_inference.utils.context import get_context


# 保留 nano-vllm 中涉及到 TP 的 Embedding 和 LMHead 类
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0  # 确保词表大小可以被 TP 大小整除，同时也避免某些 GPU 上的词表需要 padding
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size  # 每个 GPU 的词表大小
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank  # 每个 GPU 的词表起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition  # 每个 GPU 的词表结束索引
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))  # 相当于列并行
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # x: (total_tokens,) 所有的输入 token 是铺平的
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)  # mask 用于标记哪些输入索引属于当前 GPU 的词表范围 (total_tokens,)
            x = mask * (x - self.vocab_start_idx)  # x - vocab_start_idx 使得索引映射到当前 GPU 的词表范围内，mask 用于屏蔽不属于当前 GPU 的索引
        y = F.embedding(x, self.weight)  # (total_tokens, embedding_dim)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y  # 将不属于当前 GPU 的索引对应的 embedding 输出置为 0
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # x: (total_tokens, embedding_dim)
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1  # 只取每个序列的最后一个 token 的 hidden state 进行计算
            x = x[last_indices].contiguous()  # (num_seqs, embedding_dim)
        logits = F.linear(x, self.weight)  # (num_seqs, num_embeddings_per_partition) 每张卡只算出自己负责的词表部分的 logits
        if self.tp_size > 1:
            # 在 rank 0 上收集所有 GPU 的 logits，其他 rank 上的 logits 为 None
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)  # 把每张卡的 logits 收集到 rank0 的 all_logits 中
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None  # (num_seqs, num_embeddings) 在 rank0 上拼接所有 GPU 的 logits，得到完整的词表 logits
        return logits
