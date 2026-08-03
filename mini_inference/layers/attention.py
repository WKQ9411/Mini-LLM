import torch
from torch import nn

from mini_inference.kernels.store_kvcache import store_kvcache
from mini_inference.utils.context import get_context


def get_attention_backend(backend: str):
    if backend == "triton":
        from mini_inference.kernels.flash_attn_varlen_func import flash_attn_varlen_func
        from mini_inference.kernels.flash_attn_with_kvcache import flash_attn_with_kvcache
    elif backend == "flash_attn":
        try:
            from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        except ImportError as exc:
            raise ImportError(
                "attention_backend='flash_attn' requires a working flash-attn installation"
            ) from exc
    else:
        raise ValueError(f"Unsupported attention backend: {backend}")
    return flash_attn_varlen_func, flash_attn_with_kvcache


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        backend,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.prefill_attention, self.decode_attention = get_attention_backend(backend)
        self.k_cache = self.v_cache = torch.tensor([])  # 在 model_runner 中会被预分配 kv cache 替换 (num_kvcache_blocks, block_size, num_kv_heads, head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q: (num_tokens, num_heads, head_dim)
        # k, v: (num_tokens, num_kv_heads, head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():  # 预分配 kv cache 后，numel 就不为 0，开始将 k/v 存入 kv cache
            # 根据 slot_mapping，将本轮连续计算出的 k/v 分散写入预分配的 paged KV cache：
            # k/v:       (num_tokens, num_kv_heads, head_dim)
            # k/v cache: (num_kvcache_blocks, block_size, num_kv_heads, head_dim)
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache，如果没有 prefix cache，则使用连续 kv 执行 attention 计算
                k, v = k_cache, v_cache
            o = self.prefill_attention(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:  # decode
            o = self.decode_attention(q.unsqueeze(1), k_cache, v_cache,
                                      cache_seqlens=context.context_lens, block_table=context.block_tables,
                                      softmax_scale=self.scale, causal=True)
        return o
