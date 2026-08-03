import torch
import triton
import triton.language as tl


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)  # 获取当前 program 的 id
    slot = tl.load(slot_mapping_ptr + idx)  # 读取第 idx 个 token 对应的 slot mapping
    if slot == -1: 
        return  # 如果 slot 为 -1，说明该 token 不需要存入 kv cache，直接返回
    
    # key_offsets 构造这个 token 的 key 在线性内存里的地址偏移，key_stride 是 key.stride(0)，表示跨 token 的步长
    # tl.arange(0, D) 表示 token 内部 num_kv_heads * head_dim 这整段向量
    # value_offsets 同理
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # 读取这个 token 对应的 key 和 value 向量
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # 计算这个 token 对应的 slot 在 kv cache 中的区间，然后存入 kv cache 中对应的槽位
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # key/value: (N, num_kv_heads, head_dim)
    # k_cache/v_cache: (num_kvcache_blocks, block_size, num_kv_heads, head_dim)
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 一个 token 的 kv 总维度
    # 在连续的情况下，key/value 的 stride 为 (D, head_dim, 1)
    # k_cache/v_cache 的 stride 为 (block_size * D, D, head_dim, 1)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)