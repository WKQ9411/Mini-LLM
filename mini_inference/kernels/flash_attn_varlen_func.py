import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_varlen_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    block_table_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    PAGED_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 获取当前 program 的 id
    query_block = tl.program_id(0)
    query_head = tl.program_id(1)
    sequence = tl.program_id(2)


    # 通过 cu_seqlens_q 和 cu_seqlens_k 获取当前 sequence 的 query 和 key 的起始和结束位置
    query_start = tl.load(cu_seqlens_q_ptr + sequence)  # 当前 sequence 在拼接后整批 query token 里的起始位置
    query_end = tl.load(cu_seqlens_q_ptr + sequence + 1)  # 当前 sequence 在拼接后整批 query token 里的结束位置
    key_start = tl.load(cu_seqlens_k_ptr + sequence)  # 当前 sequence 在拼接后整批 cache 里的起始位置
    key_end = tl.load(cu_seqlens_k_ptr + sequence + 1)  # 当前 sequence 在拼接后整批 cache 里的结束位置
    query_length = query_end - query_start  # 当前 sequence 的 query 长度
    key_length = key_end - key_start  # 当前 sequence 的 cache 长度

    # 读取 query 的块数据
    query_block_start = query_block * BLOCK_M
    if query_block_start >= query_length:
        return  # 当前 sequence 比 batch 内最长序列短时，跳过 grid 补出的无效 query block

    query_offsets = query_block_start + tl.arange(0, BLOCK_M)  # 当前 sequence 内部的局部偏移，以该 sequence 的起点为基准 (BLOCK_M,)
    dim_offsets = tl.arange(0, BLOCK_D)  # 当前 program 处理的 head_dim 的偏移量 (BLOCK_D,)
    query_mask = query_offsets < query_length  # query token 的掩码，越界的部分为 False (BLOCK_M,)
    # query_start + query_offsets 是这 BLOCK_M 个 token 的全局索引
    # query_base 是第 sequence 个序列的第 query_block 个块里的 BLOCK_M 个 token 的第 query_head 个 head 那部分，在扁平张量 q 中的起始地址
    query_base = (query_start + query_offsets) * NUM_Q_HEADS * HEAD_DIM + query_head * HEAD_DIM  # (BLOCK_M,)
    query = tl.load(
        q_ptr + query_base[:, None] + dim_offsets[None, :],
        mask=query_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )  # (BLOCK_M, BLOCK_D) 加载 query 块，越界部分用 0 填充

    # GQA/MQA 将多个连续的 query heads 映射到一个 kv head
    queries_per_kv_head: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = query_head // queries_per_kv_head  # 当前 query head 对应的 kv head id

    # 初始化 online softmax 需要维护的状态
    row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)  # (BLOCK_M,) 记录每一行的最大值
    row_sum = tl.zeros([BLOCK_M], tl.float32)  # (BLOCK_M,) 记录每一行的归一化分母
    output_accumulator = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)  # (BLOCK_M, BLOCK_D) 记录输出的累加器
    row_has_values = tl.zeros([BLOCK_M], tl.int1)  # (BLOCK_M,) 记录每一行是否有有效值

    # 非因果时可以看到全部 key，所以 key_scan_end = key_length，因果时只能看到到当前 query block 最后一个 token 为止的历史 key
    # 对于 prefix cache，key_length - query_length 是 query 在完整 kv 序列中的起始偏移
    # + query_block_end 把 query 内局部位置映射到完整 key 轴位置
    key_scan_end = key_length
    if CAUSAL:
        query_block_end = tl.minimum(query_block_start + BLOCK_M, query_length)
        key_scan_end = tl.minimum(key_length, key_length - query_length + query_block_end)

    # 按当前 sequence 和 query block 的实际可见范围逐块遍历 kv
    key_block_start = 0
    while key_block_start < key_scan_end:
        # 计算当前 kv 块的偏移量
        key_offsets = key_block_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < key_scan_end  # 当前 kv 块的掩码，越界或超出因果边界的部分为 False (BLOCK_N,)

        if PAGED_KV:
            logical_blocks = key_offsets // CACHE_BLOCK_SIZE  # 计算每个位置所属的逻辑块 (BLOCK_N,)
            offsets_in_block = key_offsets % CACHE_BLOCK_SIZE  # 每个位置在逻辑块内的偏移量 (BLOCK_N,)
            physical_blocks = tl.load(
                block_table_ptr + sequence * NUM_BLOCKS + logical_blocks,
                mask=key_mask,
                other=0,
            )  # 根据逻辑块获取物理块的 id (BLOCK_N,)
            kv_base = physical_blocks * CACHE_BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM + offsets_in_block * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        else:
            kv_base = (key_start + key_offsets) * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

        # 加载 key 和 value 块
        key = tl.load(
            k_ptr + kv_base[None, :] + dim_offsets[:, None],
            mask=(dim_offsets[:, None] < HEAD_DIM) & key_mask[None, :],
            other=0.0,
        )  # (BLOCK_D, BLOCK_N) 加载 key 块，越界部分用 0 填充
        value = tl.load(
            v_ptr + kv_base[:, None] + dim_offsets[None, :],
            mask=key_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )  # (BLOCK_N, BLOCK_D) 加载 value 块，越界部分用 0 填充

        attention_mask = query_mask[:, None] & key_mask[None, :]  # (BLOCK_M, BLOCK_N) 计算 attention 掩码，越界部分为 False
        if CAUSAL:
            query_positions = query_offsets + key_length - query_length  # 计算 query token 在整个序列中的位置 (BLOCK_M,)
            attention_mask &= key_offsets[None, :] <= query_positions[:, None]  # 加入因果掩码

        scores = tl.dot(query, key) * softmax_scale  # (BLOCK_M, BLOCK_N) 计算注意力分数
        scores = tl.where(attention_mask, scores, -float("inf"))  # 掩码屏蔽

        # 最后一个 query block 可能包含补齐的无效行，因此需要记录每一行是否已经看到有效 key
        block_has_values = tl.max(attention_mask.to(tl.int32), axis=1) != 0  # (BLOCK_M,) 有效行为 1，无效行为 0
        new_has_values = row_has_values | block_has_values  # 历史有效标记和和当前块有效标记的并集更新
        block_max = tl.max(scores, axis=1)  # (BLOCK_M,) 计算当前块的最大值
        new_max = tl.maximum(row_max, block_max)  # (BLOCK_M,) 更新当前全局最大值
        new_max = tl.where(new_has_values, new_max, 0.0)  # 将无效行的最大值设为 0
        correction = tl.where(row_has_values, tl.exp(row_max - new_max), 0.0)  # 计算修正系数 (BLOCK_M,) 对于无效行，修正系数为 0
        probabilities = tl.where(attention_mask, tl.exp(scores - new_max[:, None]), 0.0)  # 计算当前块的 softmax 分子 (BLOCK_M, BLOCK_N) 对于无效行，概率分布为 0

        row_sum = row_sum * correction + tl.sum(probabilities, axis=1)  # (BLOCK_M,) 累加 softmax 分母
        output_accumulator *= correction[:, None]  # (BLOCK_M, BLOCK_D) 修正累加输出
        if IS_BF16:
            output_accumulator += tl.dot(probabilities.to(tl.bfloat16), value)  # (BLOCK_M, BLOCK_D) 累加本轮的输出
        else:
            output_accumulator += tl.dot(probabilities.to(tl.float16), value)  # (BLOCK_M, BLOCK_D) 累加本轮的输出
        row_max = tl.where(new_has_values, new_max, row_max)  # (BLOCK_M,) 更新当前全局最大值，对于无效行，保持原来的最大值
        row_has_values = new_has_values
        key_block_start += BLOCK_N

    denominator = tl.where(row_sum > 0.0, row_sum, 1.0)  # (BLOCK_M,) 对于无效行，分母设为 1，避免除以 0
    output = output_accumulator / denominator[:, None]   # (BLOCK_M, BLOCK_D) 归一化输出
    tl.store(
        output_ptr + query_base[:, None] + dim_offsets[None, :],  # 输出存储位置与 query 的位置对应
        output,
        mask=query_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM),
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,  # 本轮调度的最大序列长度
    max_seqlen_k: int,  # 保留用于兼容 flash-attn 接口，当前 kernel 使用 cu_seqlens_k 中的实际长度，用不到 max_seqlen_k
    softmax_scale: float | None = None,
    causal: bool = False,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    # q: (num_tokens, num_heads, head_dim)
    # k, v 是 non-paged 时: (num_tokens, num_kv_heads, head_dim)
    # k, v 是 paged 时: (num_kvcache_blocks, cache_block_size, num_kv_heads, head_dim)
    # cu_seqlens_q, cu_seqlens_k: (num_sequences + 1,)
    # block_table: (num_seqs, max_num_blocks) or None
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype
    assert q.ndim == 3 and k.shape == v.shape
    assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32

    _, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[-2]

    # 确保连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    paged_kv = block_table is not None  # 如果 block_table 不为 None，则说明 k, v 是 paged 的
    if paged_kv:
        assert k.ndim == 4 and block_table.ndim == 2
        block_table = block_table.contiguous()
        num_blocks = block_table.shape[1]
        cache_block_size = k.shape[1]
    else:
        assert k.ndim == 3
        # 非 paged 用不到的参数，做占位处理
        block_table = cu_seqlens_q
        num_blocks = cache_block_size = 1

    output = torch.empty_like(q)  # 预分配输出张量
    block_m = 16  # 一个 Triton program 一次处理多少个 query token
    block_n = 64  # 每轮加载和处理多少个 kv token，kernel 会按 kv block 循环
    block_d = triton.next_power_of_2(head_dim)  # 找到大于等于 head_dim 的最小 2 的次方
    batch_size = cu_seqlens_q.numel() - 1  # 也就是 num_sequences
    grid = (triton.cdiv(max_seqlen_q, block_m), num_q_heads, batch_size)  # 3D Grid: (num_q_blocks, num_q_heads, batch_size)

    _flash_attn_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        softmax_scale if softmax_scale is not None else head_dim**-0.5,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        NUM_BLOCKS=num_blocks,
        CACHE_BLOCK_SIZE=cache_block_size,
        PAGED_KV=paged_kv,
        CAUSAL=causal,
        IS_BF16=q.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return output
