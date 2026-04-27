from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def is_flash_attention_available() -> bool:
    return triton is not None and tl is not None and torch.cuda.is_available()


if triton is not None and tl is not None:

    @triton.jit
    def flash_attention_batched_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, Mask_ptr,
        Q_LEN, KV_LEN,
        D_QK: tl.constexpr,
        D_V: tl.constexpr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_mb, stride_mh, stride_mq, stride_mk,
        scale,
        HAS_MASK: tl.constexpr,
        MASK_IS_BOOL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D_QK: tl.constexpr,
        BLOCK_D_V: tl.constexpr,
    ):
        # 3D Grid: (num_q_blocks, batch, n_heads) 获取每个 program 的 id
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        # 偏移到当前 (batch, head) 的起始位置
        Q_ptr += pid_b * stride_qb + pid_h * stride_qh
        K_ptr += pid_b * stride_kb + pid_h * stride_kh
        V_ptr += pid_b * stride_vb + pid_h * stride_vh
        O_ptr += pid_b * stride_ob + pid_h * stride_oh

        # 计算偏移量
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
        offs_qk = tl.arange(0, BLOCK_D_QK)  # (BLOCK_D_QK,)
        offs_v = tl.arange(0, BLOCK_D_V)  # (BLOCK_D_V,)

        # 加载 Q 块
        q_ptrs = Q_ptr + offs_m[:, None] * stride_qn + offs_qk[None, :] * stride_qd  # 一个指针矩阵 (BLOCK_M, BLOCK_D_QK)
        q_mask = (offs_m[:, None] < Q_LEN) & (offs_qk[None, :] < D_QK)  # 计算掩码
        Q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)  # (BLOCK_M, BLOCK_D_QK) 加载 Q 块，越界部分用 0 填充

        # 初始化需要维护的状态，softmax 相关计算仍使用 fp32
        m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)  # (BLOCK_M,)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # (BLOCK_M,)
        O_acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)  # (BLOCK_M, BLOCK_D_V) 存储的是未归一化的累加输出

        # 遍历 K/V 块
        for j_start in range(0, KV_LEN, BLOCK_N):
            # 计算偏移量
            offs_n = j_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

            # 加载 K/V 块
            k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_qk[None, :] * stride_kd
            k_mask = (offs_n[:, None] < KV_LEN) & (offs_qk[None, :] < D_QK)
            K_block = tl.load(k_ptrs, mask=k_mask, other=0.0)  # (BLOCK_N, BLOCK_D_QK)

            v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vd
            v_mask = (offs_n[:, None] < KV_LEN) & (offs_v[None, :] < D_V)
            V_block = tl.load(v_ptrs, mask=v_mask, other=0.0)  # (BLOCK_N, BLOCK_D_V)

            # 计算 block 内的点积注意力分数
            S_block = tl.dot(Q_block, tl.trans(K_block)) * scale  # (BLOCK_M, BLOCK_N)

            # 应用 mask
            if HAS_MASK:
                mask_ptrs = (
                    Mask_ptr
                    + pid_b * stride_mb
                    + pid_h * stride_mh
                    + offs_m[:, None] * stride_mq
                    + offs_n[None, :] * stride_mk
                )
                mask_valid = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
                mask_block = tl.load(mask_ptrs, mask=mask_valid, other=0.0)
                if MASK_IS_BOOL:
                    S_block = tl.where(mask_block, S_block, float("-inf"))  # 如果是 bool mask，则在 True 的地方直接应用 -inf
                else:
                    S_block += mask_block  # 如果是数值 mask，则直接相加 

            # 越界位置设为 -inf
            kv_valid = offs_n[None, :] < KV_LEN
            q_valid = offs_m[:, None] < Q_LEN
            S_block = tl.where(q_valid & kv_valid, S_block, float("-inf"))

            m_block = tl.max(S_block, axis=1)  # 找到 block 内的最大 m (BLOCK_M,)
            m_new = tl.maximum(m_i, m_block)   # 计算当前全局最大值 (BLOCK_M,)
            alpha = tl.exp(m_i - m_new)        # 计算修正系数 (BLOCK_M,)
            P_block = tl.exp(S_block - m_new[:, None])  # 本轮的 P_block (BLOCK_M, BLOCK_N)

            l_i = l_i * alpha + tl.sum(P_block, axis=1)  # 累加 l_i (BLOCK_M,)
            O_acc = O_acc * alpha[:, None] + tl.dot(P_block.to(V_block.dtype), V_block)  # 累加 O_acc (BLOCK_M, BLOCK_D_V)
            m_i = m_new  # 更新当前最大值

        # 归一化
        # 上面 S_block = tl.where(q_valid & kv_valid, S_block, float("-inf")) 对越界行也设置为了 -inf，因此会存在 l_i 中有 0 的情况
        # 因此仅对 l_i > 0 的情况才进行归一化，等于0 的情况属于越界，直接将输出设为 0
        O_acc = tl.where(l_i[:, None] > 0.0, O_acc / l_i[:, None], 0.0)

        # 写回输出
        o_ptrs = O_ptr + offs_m[:, None] * stride_on + offs_v[None, :] * stride_od  # (BLOCK_M, BLOCK_D_V)
        o_mask = (offs_m[:, None] < Q_LEN) & (offs_v[None, :] < D_V)
        tl.store(o_ptrs, O_acc.to(O_ptr.dtype.element_ty), mask=o_mask)

else:
    flash_attention_batched_kernel = None


def _normalize_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dim() != 4:
        raise ValueError("flash_attention_forward expects a 4D attention mask")

    attention_mask = attention_mask[:, :, :q_len, :kv_len]
    if attention_mask.shape[1] == 1 and num_heads != 1:
        attention_mask = attention_mask.expand(batch_size, num_heads, q_len, kv_len)
    elif attention_mask.shape[1] != num_heads:
        raise ValueError("attention mask head dimension must be 1 or match query heads")
    return attention_mask


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    triton flash attention python 接口, 当前实现仅用于 prefill 阶段
    
    Args:
        query (torch.Tensor): (batch_size, num_heads, seq_len, head_dim)
        key (torch.Tensor): (batch_size, num_heads, seq_len, head_dim)
        value (torch.Tensor): (batch_size, num_heads, seq_len, head_dim)
        attention_mask (Optional[torch.Tensor]): (batch_size, num_heads|1, seq_len, seq_len)
        scale (Optional[float]): attention score 缩放因子
    """
    # step 1: 前置检查
    if flash_attention_batched_kernel is None:
        raise RuntimeError("Triton flash attention is not available")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be 4D tensors shaped (batch, heads, seq, dim)")
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("Triton flash attention requires CUDA tensors")

    batch_size, num_heads, q_len, qk_dim = query.shape
    _, _, kv_len, value_dim = value.shape

    attention_mask = _normalize_attention_mask(attention_mask, batch_size, num_heads, q_len, kv_len)  # None 或 (batch_size, num_heads, q_len, kv_len)
    if attention_mask is not None and not attention_mask.is_cuda:
        raise RuntimeError("attention_mask must be on CUDA when using Triton flash attention")

    if scale is None:
        scale = qk_dim ** -0.5

    # step 2: 准备 kernel 参数
    # 输出张量预分配
    output = torch.empty(
        (batch_size, num_heads, q_len, value_dim),
        device=query.device,
        dtype=query.dtype,
    )

    # 找到大于等于 qk_dim/value_dim 的最小 2 的次方，因为 triton kernel 的 block size 需要是 2 的次方
    block_d_qk = triton.next_power_of_2(qk_dim)
    block_d_v = triton.next_power_of_2(value_dim)

    # block m 和 n 分别对 Q 和 KV 的长度进行分块，固定设置为 32
    block_m = 32
    block_n = 32
    
    # 3D Grid: (num_q_blocks, batch, n_heads)
    grid = (triton.cdiv(q_len, block_m), batch_size, num_heads)

    # 设定 mask 参数
    has_mask = attention_mask is not None
    mask_is_bool = bool(has_mask and attention_mask.dtype == torch.bool)
    if has_mask:
        mask_strides = attention_mask.stride()
        mask_ptr = attention_mask
    else:
        # 此时仅作占位
        mask_strides = (0, 0, 0, 0)
        mask_ptr = output

    # step 3: 调用 kernel
    flash_attention_batched_kernel[grid](
        query, key, value, output, mask_ptr,
        q_len, kv_len,
        qk_dim, value_dim,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        mask_strides[0], mask_strides[1], mask_strides[2], mask_strides[3],
        scale,
        has_mask,
        mask_is_bool,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D_QK=block_d_qk,
        BLOCK_D_V=block_d_v,
    )
    return output
