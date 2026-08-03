import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_with_kvcache_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    output_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 获取当前 program 的 id
    sequence = tl.program_id(0)
    query_head = tl.program_id(1)

    # decode 每条序列只有一个 query token
    cache_length = tl.load(cache_seqlens_ptr + sequence)  # 获取当前 sequence 的 kv cache 长度
    dim_offsets = tl.arange(0, BLOCK_D)  # 当前 program 处理的 head_dim 的偏移量 (BLOCK_D,)
    dim_mask = dim_offsets < HEAD_DIM
    query_base = (sequence * NUM_Q_HEADS + query_head) * HEAD_DIM
    query = tl.load(
        q_ptr + query_base + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )  # (BLOCK_D,)

    queries_per_kv_head: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = query_head // queries_per_kv_head
    
    row_max = tl.full([1], -float("inf"), tl.float32)
    row_sum = tl.zeros([1], tl.float32)
    output_accumulator = tl.zeros([BLOCK_D], tl.float32)

    # CUDA Graph 会用 cache_length=0 填充未使用的 batch 位置
    if cache_length > 0:
        # cache_length 是运行时值，因此 CUDA Graph replay 时会按照当前序列的真实长度循环，
        # 避免根据固定 block_table 宽度扫描到 max_model_len 对应的最大缓存容量。
        key_block_start = 0
        while key_block_start < cache_length:
            key_offsets = key_block_start + tl.arange(0, BLOCK_N)  # 计算偏移量 (BLOCK_N,)
            key_mask = key_offsets < cache_length

            logical_blocks = key_offsets // CACHE_BLOCK_SIZE  # 计算每个位置所属的逻辑块 (BLOCK_N,)
            offsets_in_block = key_offsets % CACHE_BLOCK_SIZE  # 每个位置在逻辑块内的偏移量 (BLOCK_N,)
            physical_blocks = tl.load(
                block_table_ptr + sequence * NUM_BLOCKS + logical_blocks,
                mask=key_mask,
                other=0,
            )  # 根据逻辑块获取物理块的 id (BLOCK_N,)
            kv_base = physical_blocks * CACHE_BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM + offsets_in_block * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            # 加载 key 和 value 块
            key = tl.load(
                k_cache_ptr + kv_base[None, :] + dim_offsets[:, None],
                mask=dim_mask[:, None] & key_mask[None, :],
                other=0.0,
            )  # (BLOCK_D, BLOCK_N)
            value = tl.load(
                v_cache_ptr + kv_base[:, None] + dim_offsets[None, :],
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )  # (BLOCK_N, BLOCK_D)

            scores = tl.sum(query[:, None] * key, axis=0) * softmax_scale  # (BLOCK_N,)
            scores = tl.where(key_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)  # (1,) 计算当前块的最大值
            new_max = tl.maximum(row_max, block_max)  # (1,) 更新当前全局最大值
            correction = tl.exp(row_max - new_max)  # (1,) 计算修正系数
            probabilities = tl.where(key_mask, tl.exp(scores - new_max), 0.0)  # (BLOCK_N,) 计算当前块的 softmax 分子

            row_sum = row_sum * correction + tl.sum(probabilities, axis=0)  # (1,) 累加 softmax 分母
            output_accumulator *= correction  # (BLOCK_D,) 修正累加输出
            if IS_BF16:
                weighted_values = probabilities.to(tl.bfloat16)[:, None] * value  # (BLOCK_N, BLOCK_D) 计算加权的 value
            else:
                weighted_values = probabilities.to(tl.float16)[:, None] * value  # (BLOCK_N, BLOCK_D) 计算加权的 value
            output_accumulator += tl.sum(weighted_values, axis=0)  # (BLOCK_D,) 累加本轮的输出
            row_max = new_max
            key_block_start += BLOCK_N

    denominator = tl.where(row_sum > 0.0, row_sum, 1.0)  # (1,) 对于无效行，分母设为 1，避免除以 0
    output = output_accumulator / denominator  # (1,) 归一化输出
    tl.store(
        output_ptr + query_base + dim_offsets,  # 输出存储位置与 query 的位置对应
        output,
        mask=dim_mask,
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float | None = None,
    causal=True,
) -> torch.Tensor:
    # q: (batch_size, 1, num_heads, head_dim)
    # k/v cache: (num_kvcache_blocks, block_size, num_kv_heads, head_dim)
    # cache_seqlens: (batch_size,) 每个 batch 的 kv cache 长度
    # block_table: (num_seqs, max_num_blocks)
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k_cache.dtype == v_cache.dtype
    assert q.ndim == 4 and k_cache.ndim == 4 and k_cache.shape == v_cache.shape
    assert cache_seqlens.dtype == torch.int32 and block_table.ndim == 2

    batch_size, query_length, num_q_heads, head_dim = q.shape
    _, cache_block_size, num_kv_heads, _ = k_cache.shape
    assert query_length == 1

    # 确保连续
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    cache_seqlens = cache_seqlens.contiguous()
    block_table = block_table.contiguous()
    output = torch.empty_like(q)  # 预分配输出张量

    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    num_blocks = block_table.shape[1]
    grid = (batch_size, num_q_heads)  # 2D Grid: (batch_size, num_q_heads)
    
    _flash_attn_with_kvcache_kernel[grid](
        q,
        k_cache,
        v_cache,
        output,
        cache_seqlens,
        block_table,
        softmax_scale if softmax_scale is not None else head_dim**-0.5,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        NUM_BLOCKS=num_blocks,
        CACHE_BLOCK_SIZE=cache_block_size,
        IS_BF16=q.dtype == torch.bfloat16,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return output
