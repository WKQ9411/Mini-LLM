from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from ..rope import apply_rotary_emb, RotaryEmbedding
from ..norm import RMSNorm
from ..cache import MiniDeepSeekV4CacheLayer


# 参考代码:
# - huggingface 官方 inference 源码: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py
# - transformers 源码: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py


def dense_shared_kv_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """
    共享 kv 的 MQA dense attention

    Args:
        q:              (batch_size, num_heads, q_len, head_dim)
        kv:             (batch_size, 1, kv_len, head_dim)
        attention_mask: (batch_size, 1, q_len, kv_len)
        attn_sink:      (num_heads,)

    Returns:
        out: (batch_size, num_heads, q_len, head_dim)
    """
    dtype = q.dtype
    kv = kv.squeeze(1)  # (batch_size, kv_len, head_dim)
    scores = torch.einsum("bhsd,bkd->bhsk", q, kv) * softmax_scale  # (batch_size, num_heads, q_len, kv_len)
    scores = scores.float()
    if attention_mask is not None:
        scores = scores + attention_mask.to(scores.dtype)

    # 带 sink 的数值稳定的 softmax 计算
    sink = attn_sink.float().view(1, -1, 1)  # (1, num_heads, 1)
    max_scores = scores.max(dim=-1).values  # 每一行的最大得分 (batch_size, num_heads, q_len)
    max_scores = torch.maximum(max_scores, sink)  # (batch_size, num_heads, q_len)

    exp_scores = torch.exp(scores - max_scores.unsqueeze(-1))
    sink_exp = torch.exp(sink - max_scores)
    denom = exp_scores.sum(dim=-1) + sink_exp  # softmax 分母
    probs = (exp_scores / denom.unsqueeze(-1)).to(dtype)

    out = torch.einsum("bhsk,bkd->bhsd", probs, kv)  # (batch_size, num_heads, q_len, head_dim)
    return out


def sparse_csa_attention(
    q: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_mask: Optional[torch.Tensor],
    compressed_bias: Optional[torch.Tensor],
    compressed_index_bias: Optional[torch.Tensor],
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Per-query sparse CSA attention, 每个 query 只关注自己的 top-k 压缩 KV entry, 以及共享的 window KV entries
    window 部分和 topk 部分分别计算

    Args:
        q:               (batch_size, num_heads, q_len, head_dim)
        window_kv:       (batch_size, 1, window_kv_len, head_dim)
        compressed_kv:   (batch_size, q_len, topk, head_dim)
        window_mask:     (batch_size, 1, q_len, window_kv_len), additive mask
        compressed_bias: (batch_size, 1, q_len, topk), additive mask
        compressed_index_bias: (batch_size, q_len, topk), indexer score bias
        attn_sink:       (num_heads,)

    Returns:
        out: (batch_size, num_heads, q_len, head_dim)
    """
    dtype = q.dtype
    window_kv = window_kv.squeeze(1)  # (batch_size, window_kv_len, head_dim)

    # 分别计算 window kv 和 compressed kv 的得分
    # Window scores: (batch_size, num_heads, q_len, window_kv_len)
    window_scores = torch.einsum("bhsd,bwd->bhsw", q, window_kv) * softmax_scale
    window_scores = window_scores.float()
    if window_mask is not None:
        window_scores = window_scores + window_mask.to(window_scores.dtype)

    if compressed_kv.shape[2] > 0:
        # Compressed scores: (batch_size, num_heads, q_len, topk)
        compressed_scores = torch.einsum("bhsd,bskd->bhsk", q, compressed_kv) * softmax_scale
        compressed_scores = compressed_scores.float()
        if compressed_index_bias is not None:  # 加入 indexer 的 topk 的 score bias，使得梯度能够回传
            compressed_scores = compressed_scores + compressed_index_bias[:, None, :, :].to(compressed_scores.dtype)
        if compressed_bias is not None:
            compressed_scores = compressed_scores + compressed_bias.to(compressed_scores.dtype)
    else:
        compressed_scores = None

    # 带 sink 的数值稳定的 softmax 计算
    sink = attn_sink.float().view(1, -1, 1)  # (1, num_heads, 1)
    # 找到全局最大值
    max_scores = window_scores.max(dim=-1).values  # (batch_size, num_heads, q_len)
    if compressed_scores is not None:
        max_scores = torch.maximum(max_scores, compressed_scores.max(dim=-1).values)
    max_scores = torch.maximum(max_scores, sink)

    window_exp = torch.exp(window_scores - max_scores.unsqueeze(-1))
    sink_exp = torch.exp(sink - max_scores)
    denom = window_exp.sum(dim=-1) + sink_exp
    if compressed_scores is not None:
        compressed_exp = torch.exp(compressed_scores - max_scores.unsqueeze(-1))
        denom = denom + compressed_exp.sum(dim=-1)

    # 分别计算 window kv 和 compressed kv 的加权和
    window_probs = (window_exp / denom.unsqueeze(-1)).to(dtype)  # (batch_size, num_heads, q_len, window_kv_len)
    out = torch.einsum("bhsw,bwd->bhsd", window_probs, window_kv)  # (batch_size, num_heads, q_len, head_dim)

    if compressed_scores is not None:
        compressed_probs = (compressed_exp / denom.unsqueeze(-1)).to(dtype)  # (batch_size, num_heads, q_len, topk)
        out = out + torch.einsum("bhsk,bskd->bhsd", compressed_probs, compressed_kv)

    return out  # (batch_size, num_heads, q_len, head_dim)


class BaseCompressor(nn.Module):
    """
    BaseCompressor 是 Compressor 的基类, 分别会被用于 HCA/CSA 的主 Compressor 和 Indexer 自己的内部 Compressor
    主要功能逻辑是产生 compressed_kv
    """
    def __init__(
        self,
        hidden_size: int,
        compress_ratio: int,
        head_dim: int,
        rope_head_dim: int,
        rope_theta: float,
        max_seq_len: int,
        rms_norm_eps: float,
        overlap: bool,
        name: str,  # 用于标记缓存是 main compressor 还是 indexer compressor
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.rms_norm_eps = rms_norm_eps
        self.overlap = overlap
        self.name = name
        
        coff = 2 if overlap else 1  # CSA 的 compressor 为重叠窗口，设置为 2，HCA 则为 1
        self.position_bias = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim))  # 原文中的位置偏置参数 B
        self.wkv = nn.Linear(self.hidden_size, coff * self.head_dim, bias=False)  # KV entry 映射
        self.wgate = nn.Linear(self.hidden_size, coff * self.head_dim, bias=False)  # 压缩权重映射，每个维度都有一个权重
        self.norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(self.max_seq_len, self.rope_head_dim, self.rope_theta)
    
    def _overlap_transform(self, tensor: torch.Tensor, value=0):
        # 在 overlap 的情形下，输入的 tensor 的形状为 (batch_size, num_blocks, ratio, coff * head_dim)，其中 coff=2
        batch_size, n_blocks, _, _ = tensor.size()
        ratio, head_dim = self.compress_ratio, self.head_dim
        
        # 创建一个与输入 tensor 相同 device 和 dtype 的新张量
        new_tensor = tensor.new_full((batch_size, n_blocks, 2 * ratio, head_dim), value)  # (batch_size, n_blocks, 2*ratio, head_dim)
        
        # 将原 tensor 的后半 head_dim 部分复制给 new_tensor 的后半 ratio 部分，这相当于 C_a 或 Z_a，即本次压缩的那个 block
        new_tensor[:, :, ratio:] = tensor[:, :, :, head_dim:]
        
        # 将原 tensor 的前半 head_dim 部分复制给 new_tensor 的前半 ratio 部分，并向后错位一个 ratio，这相当于 C_b 或 Z_b，即上一个 block 的部分
        # 下标 a 的张量表征的是本轮压缩的核心信息，下标 b 的张量则表征的是更早一个 block 的历史信息
        # 注意，当 n_blocks=1 时，tensor[:, :-1, :, :head_dim] 和 new_tensor[:, 1:, :ratio] 都是空的
        # 形状均为 (batch_size, 0, ratio, head_dim)，不会执行任何复制操作，这是一个合法的边界情况
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
        return new_tensor
    
    def _build_no_pad_chunks(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        block_positions: torch.LongTensor,
        block_valid: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        """
        将无 pad 的前 usable 长度部分的 kv 和 gate 构建成可供压缩的块, 传入的 kv 和 gate 的形状为 (batch_size, usable_len, coff * head_dim)
        其中, CSA 属于重叠窗口, 其 coff 是 2, HCA 则是 1, 它们将被构建成:
        - CSA: (batch_size, n_blocks, 2*compress_ratio, head_dim)
        - HCA: (batch_size, n_blocks, compress_ratio, head_dim)
        
        Args:
            kv: (batch_size, usable_len, coff * head_dim)
            gate: (batch_size, usable_len, coff * head_dim)
            block_positions: (batch_size, n_blocks) 每个块的第一个 token 的位置索引 [直接返回, 仅用于统一接口]
            block_valid: (batch_size, n_blocks) 标记哪些块是有效的，无 pad 情况下均有效 [直接返回, 仅用于统一接口]
        
        Returns:
            chunk_kv: (batch_size, n_blocks, coff*compress_ratio, head_dim)
            chunk_gate: (batch_size, n_blocks, coff*compress_ratio, head_dim)
            block_positions: (batch_size, n_blocks)
            block_valid: (batch_size, n_blocks)
        """
        batch_size = kv.shape[0]
        n_blocks = kv.shape[1] // self.compress_ratio
        
        if n_blocks == 0:  # 当前长度不足以形成一个完整的块，返回空的 chunk_kv 和 chunk_gate
            width = 2 * self.compress_ratio if self.overlap else self.compress_ratio
            chunk_shape = (batch_size, 0, width, self.head_dim)
            return kv.new_zeros(chunk_shape), gate.new_zeros(chunk_shape), block_positions, block_valid

        position_bias = self.position_bias.to(dtype=gate.dtype)

        # (batch_size, n_blocks, compress_ratio, coff * head_dim)
        chunk_kv = kv.view(batch_size, n_blocks, self.compress_ratio, -1)
        chunk_gate = gate.view(batch_size, n_blocks, self.compress_ratio, -1) + position_bias

        # 如果是 CSA，进行重叠变换
        # 得到形状为 (batch_size, n_blocks, 2*compress_ratio, head_dim) 的 chunk_kv 和 chunk_gate
        if self.overlap:
            chunk_kv = self._overlap_transform(chunk_kv, value=0.0)
            chunk_gate = self._overlap_transform(chunk_gate, value=float("-inf"))

        return chunk_kv, chunk_gate, block_positions, block_valid

    def _build_padding_aware_chunks(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        """
        找到 batch 中所有的有效 kv 和 gate, 将每个样本的有效 token 按照 compress_ratio 划分成块

        Args:
            kv: (batch_size, seq_len, coff * head_dim)
            gate: (batch_size, seq_len, coff * head_dim)
            padding_mask: (batch_size, seq_len) 有效 token 为 1, pad token 为 0
        
        Returns:
            chunk_kv: (batch_size, max_blocks, width, head_dim)
            chunk_gate: (batch_size, max_blocks, width, head_dim)
            block_positions: (batch_size, max_blocks)
            block_valid: (batch_size, max_blocks)
        """
        batch_size = kv.shape[0]
        ratio = self.compress_ratio
        head_dim = self.head_dim
        device = kv.device

        valid = padding_mask.bool()
        logical_pos = valid.long().cumsum(dim=-1) - 1  # 计算非 padding token 的逻辑位置编号
        logical_pos = logical_pos.masked_fill(~valid, -1)  # 对于 padding token，逻辑位置编号设为 -1 (batch_size, seq_len)

        valid_len = valid.sum(dim=-1)  # 每个样本的有效长度，即非 padding token 的数量，形状为 (batch_size,)
        n_blocks = valid_len // ratio  # 每个样本实际能够形成的完整块数量，形状为 (batch_size,)
        max_blocks = int(n_blocks.max().item())  # 整个 batch 中能够形成的最大块数量

        # 初始化块位置索引和有效性标记，形状为 (batch_size, max_blocks)
        block_positions = torch.arange(max_blocks, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1) * ratio
        block_valid = torch.arange(max_blocks, device=device).unsqueeze(0) < n_blocks[:, None]

        # 没有任何一个样本能够形成完整的可压缩块
        if max_blocks == 0:
            width = 2 * ratio if self.overlap else ratio
            chunk_shape = (batch_size, 0, width, head_dim)
            return kv.new_zeros(chunk_shape), gate.new_zeros(chunk_shape), block_positions, block_valid

        b_idx, t_idx = torch.where(valid)  # 返回两个一维索引张量，形状为 (num_valid_tokens,)，分别表示 batch 维和 seq_len 维的索引
        lp = logical_pos[b_idx, t_idx]  # 获取所有有效 token 逻辑位置编号 (num_valid_tokens,)
        bid = lp // ratio  # 计算每个有效 token 所属的块索引 (num_valid_tokens,)
        off = lp % ratio  # 计算每个有效 token 在所属块内的偏移索引 (num_valid_tokens,)
        keep = bid < n_blocks[b_idx]  # 只保留能落在完整块里的有效 token (num_valid_tokens,)
        b_idx, t_idx, bid, off = b_idx[keep], t_idx[keep], bid[keep], off[keep]  # 统一按 keep 过滤
        position_bias = self.position_bias.to(dtype=gate.dtype)

        # HCA: kv/gate 直接按照块索引和块内偏移索引进行填充
        if not self.overlap:
            chunk_kv = kv.new_zeros(batch_size, max_blocks, ratio, head_dim)
            chunk_gate = gate.new_full((batch_size, max_blocks, ratio, head_dim), float("-inf"))
            chunk_kv[b_idx, bid, off] = kv[b_idx, t_idx]
            chunk_gate[b_idx, bid, off] = gate[b_idx, t_idx] + position_bias[off]
            return chunk_kv, chunk_gate, block_positions, block_valid

        # CSA: kv/gate 分别构建 a/b 两个分支，然后进行重叠变换拼接
        b_kv = kv.new_zeros(batch_size, max_blocks, ratio, head_dim)
        a_kv = kv.new_zeros(batch_size, max_blocks, ratio, head_dim)
        b_gate = gate.new_full((batch_size, max_blocks, ratio, head_dim), float("-inf"))
        a_gate = gate.new_full((batch_size, max_blocks, ratio, head_dim), float("-inf"))

        b_kv[b_idx, bid, off] = kv[b_idx, t_idx, :head_dim]
        a_kv[b_idx, bid, off] = kv[b_idx, t_idx, head_dim:]
        b_gate[b_idx, bid, off] = gate[b_idx, t_idx, :head_dim] + position_bias[off, :head_dim]
        a_gate[b_idx, bid, off] = gate[b_idx, t_idx, head_dim:] + position_bias[off, head_dim:]

        # 重叠变换拼接 a/b 分支，得到形状为 (batch_size, max_blocks, 2*ratio, head_dim) 的 chunk_kv 和 chunk_gate
        chunk_kv = kv.new_zeros(batch_size, max_blocks, 2 * ratio, head_dim)
        chunk_gate = gate.new_full((batch_size, max_blocks, 2 * ratio, head_dim), float("-inf"))
        chunk_kv[:, :, ratio:] = a_kv
        chunk_gate[:, :, ratio:] = a_gate
        chunk_kv[:, 1:, :ratio] = b_kv[:, :-1]
        chunk_gate[:, 1:, :ratio] = b_gate[:, :-1]
        return chunk_kv, chunk_gate, block_positions, block_valid
    
    def _compress(
        self, 
        chunk_kv: torch.Tensor, 
        chunk_gate: torch.Tensor, 
        block_valid: torch.BoolTensor, 
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        将 chunk_kv 和 chunk_gate 压缩成 compressed_kv, 形状为 (batch_size, n_blocks, head_dim), 其中 n_blocks 是能够形成的块数量
        会根据 block_valid 只压缩有效的块
        
        Args:
            chunk_kv: (batch_size, n_blocks, width, head_dim)
            chunk_gate: (batch_size, n_blocks, width, head_dim)
            block_valid: (batch_size, n_blocks)
            dtype: torch.dtype 输出数据类型
        """
        # 如果不存在可压缩块，直接返回空的 compressed_kv
        if chunk_kv.shape[1] == 0:
            return chunk_kv.new_zeros(chunk_kv.shape[0], 0, self.head_dim)

        valid = block_valid[:, :, None, None]  # (batch_size, n_blocks, 1, 1)，用于扩展 block_valid 以匹配 chunk_gate 的形状
        safe_gate = torch.where(valid, chunk_gate.float(), torch.zeros_like(chunk_gate.float()))  # 将无效 block 的 gate 填充为 0，避免在 softmax 中出现 NaN
        weights = safe_gate.softmax(dim=2, dtype=torch.float32).to(dtype)  # (batch_size, n_blocks, width, head_dim)，对块内的 gate 进行 softmax，得到权重
        compressed = (chunk_kv * weights).sum(dim=2).to(dtype)  # (batch_size, n_blocks, head_dim)，根据权重对块内的 kv 进行加权求和，得到压缩后的 kv entry
        compressed = compressed.masked_fill(~block_valid[:, :, None], 0.0)  # 将无效 block 的 compressed kv entry 填充为 0
        return self.norm(compressed)
    
    def _apply_rope(self, compressed: torch.Tensor, block_positions: torch.LongTensor) -> torch.Tensor:
        """
        对 compressed kv 应用 RoPE, Compressor 需要有自己的 RoPE 模块, 
        因为在解码时, 主模型的输入仅有一个 token, 相应产生的 position_embeddings 也只能有一个位置
        但是在压缩时, 需要该 token 所对应的 block 的第一个 token 的位置 的 position_embeddings 来进行 RoPE 计算
        此时主模型的 RoPE 模块已无法提供, 因此 Compressor 需要自己的 RoPE 模块来提供正确的 position_embeddings
        """
        # 如果为空，无需应用 RoPE，直接返回
        if compressed.shape[1] == 0:
            return compressed
        
        position_embeddings = self.rotary_emb(compressed, block_positions)
        compressed = compressed.unsqueeze(1)  # 添加头维度以兼容本项目的 RoPE 实现，形状变为 (batch_size, 1, n_blocks, head_dim)
        
        nope = compressed[..., : self.nope_head_dim]
        rope = compressed[..., self.nope_head_dim :]
        rope = apply_rotary_emb(rope, position_embeddings)
        return torch.cat([nope, rope], dim=-1).squeeze(1)  # 去掉头维度，返回形状为 (batch_size, n_blocks, head_dim)
    
    def _make_block_bias(
        self,
        block_valid: torch.BoolTensor,
        position_ids: torch.LongTensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        为压缩后的 kv entry 生成可见性因果偏置, 让每个 query 只能看到已经闭合的历史压缩块, 同时结合 block_valid
        这里的逻辑类似于 pad mask 和 causal mask 的结合
        最终返回的 block_bias 形状为 (batch_size, 1, q_len, n_blocks), 其中 n_blocks 是能够形成的块的最大数量
        
        Args:
            block_valid: (batch_size, n_blocks) 标记哪些块是有效的, 类似于 attention_mask, 指的是所有的历史压缩块的掩码标记, 而不是仅本轮 decode 的那一部分
            position_ids: (batch_size, q_len) 每个 query 的位置索引
            dtype: 输出数据类型
        """
        batch_size, n_blocks = block_valid.shape
        q_len = position_ids.shape[1]
        if n_blocks == 0:
            return torch.empty(batch_size, 1, q_len, 0, dtype=dtype, device=position_ids.device)  # 返回空

        entry_idx = torch.arange(n_blocks, device=position_ids.device)  # (n_blocks,) 每个压缩块的索引
        # 每个 query 可见的压缩块数量上限
        # 该逻辑的可视化参考：https://github.com/huggingface/transformers/pull/45892
        # 例如 compress_ratio=4 时，position_ids=0,1,2 的 query +1 后 // 4 为 0，它们看不到任何压缩块，因为它们所处的压缩块还未闭合
        # 但对 position_ids=3 的 query 来说，+1 后 // 4 为 1，它可以看到第 0 个压缩块，因为它所处的压缩块已经闭合了
        causal_threshold = (position_ids + 1) // self.compress_ratio  # (batch_size, q_len)
        visible = entry_idx[None, None, :] < causal_threshold[:, :, None]  # (batch_size, q_len, n_blocks) 每个 query 可见哪些压缩块
        visible = visible & block_valid[:, None, :]  # 结合 block_valid，确保 query 只能看到有效的压缩块
        bias = torch.where(
            visible[:, None, :, :],
            torch.zeros((), dtype=dtype, device=position_ids.device),
            torch.full((), torch.finfo(dtype).min, dtype=dtype, device=position_ids.device),
        )  # 对于可见的压缩块，偏置为 0；对于不可见的压缩块，偏置为 -inf，使其在 softmax 后对应的权重为 0
        return bias
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        layer_idx: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Returns:
            compressed_kv: (batch_size, 1, n_blocks, head_dim) 压缩后的 kv entry, 其中 n_blocks 是所有的历史压缩块长度
            block_bias:    (batch_size, 1, q_len, n_blocks)  每个 query 可见哪些 compressed kv
            valid_mask:    (batch_size, n_blocks)  标记哪些块是有效的
        """
        batch_size, seq_len, _ = hidden_states.size()
        dtype = hidden_states.dtype
        cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
        
        # 获取 kv 和 gate，分别对应论文中的 C 和 Z
        kv = self.wkv(hidden_states)      # (batch_size, seq_len, coff * head_dim)
        gate = self.wgate(hidden_states)  # (batch_size, seq_len, coff * head_dim)
        
        # 获取 kv 和 gate 中能够被压缩的部分
        # Case 1: 不使用缓存的无状态情况，通常是训练的情况，尤其是 SFT 时，要考虑 padding
        # NOTE: 截至本项目实现当前逻辑时，inference 源码和 transformers 源码中似乎在 compressor 中没有处理 pad token 的逻辑
        # 对于训练，尤其是 SFT 而言，不同样本有效长度不一致，进行 compress 时需要考虑 pad token，只压缩有效长度部分
        # 因此，本项目这里加入了考虑 pad token 的逻辑，以确保正确的 compressed kv 形成和训练
        # 相关问题我已提 issue：https://github.com/huggingface/transformers/issues/45938 后续可能会更改，但本项目先按照自己的逻辑进行实现
        if cache_layer is None:
            if padding_mask is None:  # 无 pad 情况，通常是预训练
                usable = (seq_len // self.compress_ratio) * self.compress_ratio  # 计算能够被 compress_ratio 整除从而进行压缩的部分
                # 截取出可被压缩的部分
                flat_kv = kv[:, :usable]
                flat_gate = gate[:, :usable]
                # 计算可被压缩的块数量、每个块的位置索引和有效性标记，其中，每个块的位置索引使用的是每个块的第一个 token 的位置索引
                # 如果 n_blocks 为 0，则 block_positions 和 block_valid 都是空的，但这也是合法的边界情况
                n_blocks = usable // self.compress_ratio
                block_positions = torch.arange(0, usable, self.compress_ratio, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_blocks)
                block_valid = torch.ones(batch_size, n_blocks, dtype=torch.bool, device=hidden_states.device)  # (batch_size, n_blocks) 标记哪些块是有效的，无 pad 情况下均有效
                chunk_kv, chunk_gate, block_positions, block_valid = self._build_no_pad_chunks(flat_kv, flat_gate, block_positions, block_valid)
            else:  # 有 pad 情况
                chunk_kv, chunk_gate, block_positions, block_valid = self._build_padding_aware_chunks(kv, gate, padding_mask)
            
            compressed = self._compress(chunk_kv, chunk_gate, block_valid, dtype)  # (batch_size, n_blocks, head_dim)
            # DeepSeekV4 在 huggingface 上发布的 inference/model.py 源码中，通过 self.compress_ratio 进行划分
            # 当使用压缩时，CSA 和 HCA 的 RoPE theta 统一为 40000，不使用压缩注意力的纯滑动窗口层则使用 10000
            compressed = self._apply_rope(compressed, block_positions)  # (batch_size, n_blocks, head_dim)
            block_bias = self._make_block_bias(block_valid, position_ids, dtype)
            return compressed.unsqueeze(1), block_bias, block_valid
        
        # Case 2: 使用缓存的有状态情况，通常是推理的情况，本项目暂时只考虑单样本无 pad 的推理
        # 对于左 pad 的批量推理，构建 compressed kv 并维护 cache 逻辑会比较复杂，暂不考虑
        if padding_mask is not None and not bool(padding_mask.all()):
            raise NotImplementedError(
                "The compression-cache path currently supports only single-batch, no-padding inference. "
                "Please use no-padding inputs, or disable cache for padded scenarios."
            )
        
        flat_kv, flat_gate, block_positions, block_valid = cache_layer.update_compressor_buffer(
            self.name, self.compress_ratio, kv, gate, self.overlap
        )  # 存入 buffer，当有新的可压缩部分时，会返回非空结果
        chunk_kv, chunk_gate, block_positions, block_valid = self._build_no_pad_chunks(
            flat_kv, flat_gate, block_positions, block_valid
        )  # 本项目在推理时使用无 pad 的块构建逻辑
        new_compressed = self._compress(chunk_kv, chunk_gate, block_valid, dtype)  # 这里是新形成的压缩块 (batch_size, n_blocks, head_dim)
        new_compressed = self._apply_rope(new_compressed, block_positions)
        
        # 对于 CSA，由于使用了重叠窗口，第一个 block 只是用于提供上下文信息，从而正确的进行 overlap transform
        # 但它应该被排除，不进入到 compressed kv 的更新和 block bias 的计算中，因此这里通过 block_valid.any(dim=0) 来排除掉第一个 block
        emit_mask = block_valid.any(dim=0)
        new_compressed = new_compressed[:, emit_mask]
        block_valid = block_valid[:, emit_mask]
        
        all_compressed, all_valid = cache_layer.update_compressor_states(self.name, new_compressed, block_valid)  # 这里返回的是所有历史压缩块和所有可见性掩码
        block_bias = self._make_block_bias(all_valid, position_ids, dtype)
        
        # 对上述的逻辑在 CSA 的情况下做一个例子说明:
        # -------------------------------
        # prefill 阶段:
        # 1. 假设一次拿到 n*ratio 个 token: [block0, block1, block2, ..., block_n-1]
        # update_compressor_buffer 初次没有 context_kv, 所以返回长度就是: n*ratio
        # 2. 然后进入 _build_no_pad_chunks 后走 CSA overlap transform
        # 对于 block0:
        #   prev = padding: kv=0, score=-inf
        #   curr = block0
        # 所以第一个 compressed block 的上一块是人为 padding 的
        # 对于 block1:
        #   prev = block0
        #   curr = block1
        # 对于 block2:
        #   prev = block1
        #   curr = block2
        # ...
        # --------------------------------
        # Decode 阶段形成新 block 时:
        # 1. 假设之前已经有 block0, 现在 decode 累积出了 block1
        # 那么 update_compressor_buffer(overlap=True) 会返回: block0 + block1, 也就是长度 2*ratio
        # 2. 进入 _build_no_pad_chunks 和 overlap transform 后, 会产生两个 compressed 候选:
        #   compressed_from_block0
        #   compressed_from_block1
        # 其中:
        #   compressed_from_block0: prev = padding, curr = block0
        #   compressed_from_block1: prev = block0, curr = block1
        # 这个 compressed_from_block0 只是为了让 transform 的序列结构完整, 并不是本轮真正要新增的压缩块
        # 所以后面用:
        #   emit_mask = block_valid.any(dim=0)
        #   new_compressed = new_compressed[:, emit_mask]
        # 把 compressed_from_block0 排除，只保留 compressed_from_block1 写入 compressed cache
        
        return all_compressed.unsqueeze(1), block_bias, all_valid
    
    
class Indexer(nn.Module):
    training_exploration_slots = 4

    def __init__(
        self, 
        hidden_size: int,
        compress_ratio: int,
        index_num_attention_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.compress_ratio = compress_ratio
        self.index_num_attention_heads = index_num_attention_heads
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = index_head_dim - rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.max_seq_len = max_seq_len
        self.softmax_scale = self.index_head_dim ** -0.5
        
        self.wq_b = nn.Linear(self.q_lora_rank, self.index_num_attention_heads * self.index_head_dim, bias=False)  # 用于产生 indexer query 的上投影矩阵
        self.weights_proj = nn.Linear(self.hidden_size, self.index_num_attention_heads, bias=False)  # 用于得到每个 indexer head 的权重
        self.rotary_emb = RotaryEmbedding(self.max_seq_len, self.rope_head_dim, rope_theta)

        # indexer 也有自己的 compressor
        self.compressor = BaseCompressor(
            hidden_size=hidden_size,
            compress_ratio=compress_ratio,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            rms_norm_eps=rms_norm_eps,
            overlap=True,
            name="indexer",
        )
    
    def _apply_query_rope(self, q: torch.Tensor, position_ids: torch.LongTensor):
        position_embeddings = self.rotary_emb(q, position_ids)
        q_nope = q[..., :self.nope_head_dim]
        q_rope = q[..., self.nope_head_dim:].transpose(1, 2)  # (batch_size, index_num_attention_heads, seq_len, rope_head_dim)
        q_rope = apply_rotary_emb(q_rope, position_embeddings).transpose(1, 2)
        return torch.cat([q_nope, q_rope], dim=-1)

    @staticmethod
    def _select_topk_with_exploration(
        scores: torch.Tensor,
        k: int,
        exploration_slots: int,
        training: bool,
        random_scores: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        选择 topk compressed blocks, 在训练时, 从 topk 中保留一定数量的随机可见候选项, 
        这样会使得那些非 topk 的块也一定程度上能够获取梯度,
        在推理时, 则取消随机可见, 全部来源于 topk
        """
        if not training or exploration_slots <= 0:  # 推理时全部采用 topk
            return torch.topk(scores, k=k, dim=-1)

        exploration_slots = min(exploration_slots, k)  # 随机探索槽位
        exploit_slots = k - exploration_slots  # 除去随机探索后剩余的真实 topk 槽位

        # 选出预留探索槽位之后的真实 topk
        if exploit_slots > 0:
            exploit_scores, exploit_indices = torch.topk(scores, k=exploit_slots, dim=-1)  # (batch_size, q_len, exploit_slots)
            candidate_scores = scores.scatter(
                dim=-1,
                index=exploit_indices,
                src=torch.full_like(exploit_scores, float("-inf")),
            )  # 将已选的分数设置为 -inf，这样后续 exploration 的选择就不会重复
        else:
            exploit_scores = scores.new_empty(scores.shape[:-1] + (0,))  # (batch_size, q_len, 0)
            exploit_indices = torch.empty(scores.shape[:-1] + (0,), dtype=torch.long, device=scores.device)  # (batch_size, q_len, 0)
            candidate_scores = scores

        # exploration 使用均匀分布的 random_scores 进行选择
        if random_scores is None:
            random_scores = torch.rand(scores.shape, dtype=torch.float32, device=scores.device)
        else:
            random_scores = random_scores.to(device=scores.device, dtype=torch.float32)

        # 将未选择(或可选择)的部分标记为 True，已选择(或不可选择)的部分为 False
        # 因为传入的 scores 本身可能已经包含 -inf, 这代表本就不能选中的块
        candidate_mask = torch.isfinite(candidate_scores)  
        exploration_random_scores = random_scores.masked_fill(~candidate_mask, float("-inf"))  # 已选择(或不可选择)部分对应的随机 score 设置为 -inf
        _, exploration_indices = torch.topk(exploration_random_scores, k=exploration_slots, dim=-1)  # (batch_size, q_len, exploration_slots)
        exploration_scores = candidate_scores.gather(dim=-1, index=exploration_indices)  # eploration 的分数从原始分数中取

        # 形成最终的 topk
        topk_scores = torch.cat([exploit_scores, exploration_scores], dim=-1)
        topk_indices = torch.cat([exploit_indices, exploration_indices], dim=-1)
        return topk_scores, topk_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        layer_idx: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """
        返回每个 indexer query 对应的 topk compressed kv 索引和可导 score bias
        
        NOTE: DeepSeekV4 论文中关于 indexer 的训练提到:
        The training starts with a sequence length of 4K, and we gradually extend the training sequence length to 16K, 64K, and 1M. 
        As for the setups of sparse attention, we first warmup the model with dense attention for the first 1T tokens, 
        and introduce sparse attention at the sequence length of 64K and keep sparse attention during the rest of the training. 
        When introducing attention sparsity, we first set a short stage to warm up the lightning indexer in CSA, 
        and then train the model with sparse attention for most of the training.
        
        DeepSeekV3.2 介绍 DSA 也提到: DSA 由 lightning indexer 和 fine-grained token selection 组成, 
        训练分为两个阶段, 第一阶段: Dense Warm-up, 只训练 indexer, 保持 dense attention, 冻结除 indexer 外的所有参数, 目的是 align indexer outputs with main attention distribution
        第二阶段: Sparse Training, main model 用 LM loss 适应 sparse attention, indexer 继续用 KL loss 对齐 main attention distribution
        
        在本项目实现的 mini deepseekv4 indexer 中, 由于单独的 topk 无法回传梯度, 同时本项目为了固化和复用训练代码逻辑, 暂时不像原始论文那样分阶段训练, 
        这里采用一个简单的方法, 直接从头训练 indexer, 将 topk 选中的 kv entries 的 score 作为 bias 加入到 core attention 中, 
        这样 topk 选中的部分就可以回传梯度, 它的弊端是未选中的部分无法获取梯度, 
        这类似于 MoE gate 的 routing, 通过 weight 来回传梯度, 但是 MoE 通过一系列负载均衡措施, 会训练的更稳定一些
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # indexer query
        q = self.wq_b(qr).view(batch_size, seq_len, self.index_num_attention_heads, self.index_head_dim)
        q = self._apply_query_rope(q, position_ids)  # (batch_size, seq_len, index_num_attention_heads, index_head_dim)
        
        # indexer compressed kv
        index_kv, _, valid_mask = self.compressor(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            padding_mask=padding_mask,
        )  # index_kv 形状为 (batch_size, 1, n_blocks, index_head_dim)，其中 n_blocks 是所有历史压缩块的长度
        index_kv = index_kv.squeeze(1)  # (batch_size, n_blocks, index_head_dim)
        n_blocks = index_kv.size(1)
        if n_blocks == 0 or self.index_topk == 0:
            empty_indices = torch.empty(batch_size, seq_len, 0, dtype=torch.long, device=hidden_states.device)
            empty_bias = torch.empty(batch_size, seq_len, 0, dtype=torch.float32, device=hidden_states.device)
            return empty_indices, empty_bias
        
        scores = torch.einsum("bqhd,bnd->bqhn", q, index_kv) * self.softmax_scale  # (batch_size, seq_len, index_num_attention_heads, n_blocks)
        scores = scores.float()
        # 由于后续需要沿 indexer head 对 socre 求和，假设每个 head 的 score 方差为 σ²，对 N 个独立的 head 求和后，方差为 N*σ²
        # 因此这里进一步乘上缩放因子 N^(-0.5)，从而使求和后的 score 方差回到 σ²，保持数值稳定性
        weights = self.weights_proj(hidden_states).float() * (self.index_num_attention_heads ** -0.5)  # (batch_size, seq_len, index_num_attention_heads)
        scores = (scores.relu() * weights[:, :, :, None]).sum(dim=2)  # 沿 indexer head 维度加权求和 (batch_size, seq_len, n_blocks)

        # 这里的逻辑和 BaseCompressor 的 _make_block_bias 类似
        # 都是为了生成可见性掩码，让每个 query 只能看到已经闭合的历史压缩块，同时应用 valid_mask
        entry_idx = torch.arange(n_blocks, device=hidden_states.device)
        threshold = (position_ids + 1) // self.compress_ratio
        visible = entry_idx[None, None, :] < threshold[:, :, None]
        visible = visible & valid_mask[:, None, :]
        scores = scores.masked_fill(~visible, float("-inf"))  # 把不可见的压缩块对应的 score 填充为 -inf，使其在后续的 topk 中不会被选中

        k = min(self.index_topk, n_blocks)  # 取 topk 和 n_blocks 中的较小值
        # 注意这里，scores 中可能存在 -inf
        # 对于 scores 中开头的几个 query，例如 query 0，因为它看不到任何 compressed kv，所以它的 scores 全是 -inf
        # 即 (batch_size, seq_len, topk) 在 seq_len=0 的那一行全是 -inf
        # 此外，也可能出现 k 大于可见压缩块数量的情况，此时也会选进来 -inf 的 score
        # 因此，这里还需要对 topk 的结果进行一次过滤，把那些 score 是 -inf 的索引替换为 -1，表示无效索引
        topk_scores, topk_indices = self._select_topk_with_exploration(
            scores=scores,
            k=k,
            exploration_slots=self.training_exploration_slots,
            training=self.training,
        )
        valid = torch.isfinite(topk_scores)  # (batch_size, seq_len, topk)

        # 计算归一化的 score bias，防止数值过大，加入 core attention 后导致不稳定
        safe_scores = torch.where(valid, topk_scores, torch.zeros_like(topk_scores))  # 无效位置的 score 填充为 0
        valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1)  # 每个 query 实际选中的有效压缩块数量，最小为 1，避免除以 0
        mean = safe_scores.sum(dim=-1, keepdim=True) / valid_count  # 计算有效 score 的均值
        centered = torch.where(valid, topk_scores - mean, torch.zeros_like(topk_scores))
        var = centered.square().sum(dim=-1, keepdim=True) / valid_count  # 计算有效 score 的方差
        topk_score_bias = (centered * torch.rsqrt(var.clamp_min(1e-6))).clamp(-5.0, 5.0)  # 归一化并裁剪 score bias，保持数值稳定性
        topk_score_bias = topk_score_bias.masked_fill(~valid, 0.0)  # 将无效位置的 score bias 填充为 0

        topk_indices = torch.where(valid, topk_indices, torch.full_like(topk_indices, -1))
        return topk_indices, topk_score_bias  # (batch_size, seq_len, topk)
        

class Compressor(BaseCompressor):
    def __init__(
        self, 
        hidden_size: int,
        compress_ratio: int, 
        head_dim: int, 
        rope_head_dim: int,
        rope_theta: float,
        max_seq_len: int,
        rms_norm_eps: float,
        layer_type: str,
        index_num_attention_heads: int,
        index_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
    ):
        overlap = layer_type == "compressed_sparse_attention"  # 如果是 CSA，使用重叠压缩
        super().__init__(
            hidden_size=hidden_size,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            rms_norm_eps=rms_norm_eps,
            overlap=overlap,
            name="compressor",
        )
        self.layer_type = layer_type
        self.indexer = None
        if overlap:
            # CSA 需要初始化 indexer
            self.indexer = Indexer(
                hidden_size=hidden_size,
                compress_ratio=compress_ratio,
                index_num_attention_heads=index_num_attention_heads,
                index_head_dim=index_head_dim,
                rope_head_dim=rope_head_dim,
                index_topk=index_topk,
                q_lora_rank=q_lora_rank,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
            )
    
    @staticmethod
    def _gather_csa_topk(
        compressed_kv: torch.Tensor,
        topk_indices: torch.LongTensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, Optional[torch.Tensor]]:
        """
        稀疏 CSA gather, 每个 query 只保留自己的 top-k compressed kv

        Args:
            compressed_kv: (batch_size, 1, n_blocks, head_dim)
            topk_indices:  (batch_size, q_len, topk), -1 表示无效索引

        Returns:
            gathered:   (batch_size, q_len, topk, head_dim)
            block_bias: (batch_size, 1, q_len, topk)
        """
        batch_size, _, n_blocks, head_dim = compressed_kv.shape
        q_len = topk_indices.shape[1]
        topk = topk_indices.shape[2]
        device = compressed_kv.device

        if topk == 0 or n_blocks == 0:
            return (
                compressed_kv.new_zeros(batch_size, q_len, 0, head_dim),
                torch.empty(batch_size, 1, q_len, 0, dtype=dtype, device=device),
            )

        # compressed_kv 的 dim 1 本来的含义注意力头，现在 expand 成 q_len，用于 gather 每个 query 各自的 top-k compressed kv
        source = compressed_kv.expand(batch_size, q_len, n_blocks, head_dim)
        # 将 topk_indices 中的 -1 替换为 0，得到一个合法的索引张量，并扩展成 (batch_size, q_len, topk, head_dim) 以用于 gather
        gather_idx = topk_indices.clamp_min(0).unsqueeze(-1).expand(batch_size, q_len, topk, head_dim)
        # 这会将无效索引 -1 位置的 compressed kv 也 gather 出来，但会在后续将其填充为 0
        gathered = torch.gather(source, dim=2, index=gather_idx)

        valid = topk_indices >= 0  # -1 的无效索引位置为 False，其他位置为 True，形状为 (batch_size, q_len, topk)
        gathered = gathered.masked_fill(~valid[..., None], 0.0)  # 将无效索引位置的 gathered kv entry 填充为 0 (batch_size, q_len, topk, head_dim)
        block_bias = torch.where(
            valid[:, None, :, :],
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
        )  # 这里的 topk 已经是在 indexer 中进行了因果过滤，因此直接根据 valid 来生成 block_bias 即可
        return gathered, block_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        layer_idx: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CSA 返回的 topk 是 per-query 的, 因此每个 query 对应不同的 compressed kv entry
        HCA 则是所有 query 共享同一套 compressed kv entry
        
        Returns:
            HCA:
                compressed_kv: (batch_size, 1, n_blocks, head_dim)
                block_bias:    (batch_size, 1, q_len, n_blocks)
                is_sparse:     False
                index_bias:    None
            CSA:
                compressed_kv: (batch_size, q_len, topk, head_dim)
                block_bias:    (batch_size, 1, q_len, topk)
                is_sparse:     True
                index_bias:    (batch_size, q_len, topk)
        """
        # BaseCompressor 的前向，返回所有的 compressed kv 和对应的 block_bias
        all_compressed, hca_bias, _ = super().forward(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            padding_mask=padding_mask,
        )

        # HCA 的返回
        if self.indexer is None:
            return all_compressed, hca_bias, False, None

        # CSA 的返回
        topk_indices, topk_score_bias = self.indexer(
            hidden_states=hidden_states,
            qr=qr,
            position_ids=position_ids,
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            padding_mask=padding_mask,
        )
        gathered, block_bias = self._gather_csa_topk(all_compressed, topk_indices, hidden_states.dtype)
        return gathered, block_bias, True, topk_score_bias


class DeepSeekV4Attention(nn.Module):
    """
    CSA (Compressed Sparse Attention) 与 HCA (Heavily Compressed Attention) 均通过本类统一实现, DeepSeekV4 版本

    Args:
        layer_idx (int): 层索引
        layer_types (Tuple[str]): 每层的类型列表, 包括 "sliding_attention"、"compressed_sparse_attention" 和 "heavily_compressed_attention"
        hidden_size (int): 隐状态维度
        num_attention_heads (int): 注意力头数
        index_num_attention_heads (int): indexer 的注意力头数
        q_lora_rank (int): query 的下投影维度
        o_lora_rank (int): output 的分组投影维度
        head_dim (int): 每个头的维度
        index_head_dim (int): indexer 每个头的维度
        rope_head_dim (int): 带 RoPE 的维度
        o_groups (int): 输出分组数
        window_size (int): 滑动窗口大小
        compress_ratios (dict): 长度压缩比例, 包含每种类型对应的压缩比例，例如 {"sliding_attention": 0, "compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
        rms_norm_eps (float): RMSNorm 的 epsilon
        index_topk (int): indexer 选取的 top-k 值
    """
    def __init__(
        self, 
        layer_idx: int, 
        layer_types: Tuple[str],
        hidden_size: int,
        num_attention_heads: int,
        index_num_attention_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        index_head_dim: int,
        rope_head_dim: int,
        compress_rope_theta: float,
        o_groups: int,
        window_size: int,
        compress_ratios: dict,
        rms_norm_eps: float,
        index_topk: int,
        index_score_bias_alpha: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_types[layer_idx]
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.index_num_attention_heads = index_num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.compress_rope_theta = compress_rope_theta
        self.o_groups = o_groups
        self.window_size = window_size
        self.compress_ratio = compress_ratios[self.layer_type]
        self.rms_norm_eps = rms_norm_eps
        self.index_topk = index_topk
        self.index_score_bias_alpha = index_score_bias_alpha
        self.max_seq_len = max_seq_len

        self.attn_sink = nn.Parameter(torch.empty(self.num_attention_heads, dtype=torch.float32))  # 每个头均有一个 attention sink 参数
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)  # Q 的下投影矩阵
        self.q_norm = RMSNorm(self.q_lora_rank, self.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_attention_heads * self.head_dim, bias=False)  # Q 的上投影矩阵
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)  # KV 的投影矩阵，这里是 MQA，因此只有单头
        self.kv_norm = RMSNorm(self.head_dim, self.rms_norm_eps)
        self.wo_a = nn.Linear(self.num_attention_heads * self.head_dim // self.o_groups, self.o_groups * o_lora_rank, bias=False)  # O 的分组投影矩阵
        self.wo_b = nn.Linear(self.o_groups * o_lora_rank, self.hidden_size, bias=False)  # O 的最终输出投影
        self.scaling = self.head_dim ** -0.5  # 注意力缩放因子

        if self.layer_type != "sliding_attention":
            self.compressor = Compressor(
                hidden_size=hidden_size,
                compress_ratio=self.compress_ratio,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                rope_theta=compress_rope_theta,
                max_seq_len=max_seq_len,
                rms_norm_eps=rms_norm_eps,
                layer_type=self.layer_type,
                index_num_attention_heads=index_num_attention_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                q_lora_rank=q_lora_rank,
            )
        else:
            self.compressor = None

    def forward(
        self, 
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # 4D causal/window mask
        padding_mask: Optional[torch.Tensor] = None,    # 2D raw padding mask
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.size()
        
        # step 1. core attention query
        qr = q = self.q_norm(self.wq_a(hidden_states))  # qr 和 q 当前指向同一个潜在向量，qr 用于产生 indexer query，(batch_size, seq_len, q_lora_rank)
        q = self.wq_b(q).unflatten(-1, (self.num_attention_heads, self.head_dim)).transpose(1, 2)  # (batch_size, num_attention_heads, seq_len, head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.rms_norm_eps)  # 对每个 head 执行的 RMSNorm（无可学习参数）
        # NOTE: 本项目为了在不同模型中复用 apply_rotary_emb 函数，保持了 RoPE 的计算方式，因此这里需要一定转换来适配 RoPE 的输入格式
        q_nope = q[..., :-self.rope_head_dim]  # 不带 RoPE 的部分直接参与后续计算 (batch_size, num_attention_heads, seq_len, nope_head_dim)
        q_rope = q[..., -self.rope_head_dim:]  # 部分 RoPE (batch_size, num_attention_heads, seq_len, rope_head_dim)
        q_rope = apply_rotary_emb(q_rope, position_embeddings)  # 这里的 position_embeddings 的 dim 需要与 rope_head_dim 一致
        q = torch.cat([q_nope, q_rope], dim=-1)  # 重新拼接 (batch_size, num_attention_heads, seq_len, head_dim)

        # step 2. core attention kv
        kv = self.kv_norm(self.wkv(hidden_states)).unsqueeze(1)  # kv 只有一个头 (batch_size, 1, seq_len, head_dim)
        kv_nope = kv[..., :-self.rope_head_dim]  # 不带 RoPE 的部分 (batch_size, 1, seq_len, nope_head_dim)
        kv_rope = kv[..., -self.rope_head_dim:]  # 部分 RoPE (batch_size, 1, seq_len, rope_head_dim)
        kv_rope = apply_rotary_emb(kv_rope, position_embeddings)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)  # 重新拼接 (batch_size, 1, seq_len, head_dim)
        
        # 由于 kv entry 同时充当了 key 和 value，这里为了兼容 past_key_values 的签名，将 kv 同时传入 key_states 和 value_states
        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]
        
        # NOTE: DeepSeekV4 在 huggingface 上发布的 inference/model.py 源码是通过构造 topk_idxs 的方式来选择应当关注的 token 的
        # 此外，inference/model.py 源码采用的是预分配的 register_buffer kv_cache，滑动窗口使用了环形缓冲逻辑
        # transformers 的 DeepSeekV4 实现中，则是通过内部接口 create_sliding_window_causal_mask 来生成滑动窗口 mask 的
        # 缓存通过 Cache 类实现，滑动窗口直接截取最新的 window_size 长度的 kv，逻辑更加直观
        # 本项目采用 transformers 的方式，直接通过 create_sliding_window_causal_mask 构造 mask
        # attention_mask 是事先由 create_sliding_window_causal_mask 构造好的纯滑动窗口 mask，形状为 (batch_size, 1, q_len, kv_len)
        # 因此，在拼接了 compressed_kv 之后，需要对 attention_mask 在 kv_len 维度上进行相应的扩充，使其与新的 kv 长度对齐
        # 关于拼接的 mask，可以参考 https://github.com/huggingface/transformers/pull/45892
        # NOTE: 在 DeepSeekV4 中，并没有要求超出了 window 的部分才进行 compress
        # 例如，假设 seq_len=130，window_size=128，compress_ratio=128
        # 那么对于位置为 129 的 query，所能看到的 window kv 是 [2, 129]，所能看到的 compressed kv 是 [0, 127] 压缩后的 kv entry
        # 因此，这中间的历史 token 信息是可以有重叠的，这样的逻辑实现起来也更简单一些
        
        # step 3. attention
        if self.compressor is None:  # 纯滑动窗口注意力，直接执行密集注意力计算
            o = dense_shared_kv_attention(
                q=q,
                kv=kv,
                attention_mask=attention_mask,
                attn_sink=self.attn_sink,
                softmax_scale=self.scaling,
            )
        else:  # HCA/CSA
            compressed_kv, block_bias, is_sparse, compressed_index_bias = self.compressor(
                hidden_states=hidden_states,
                qr=qr,
                position_ids=position_ids,
                past_key_values=past_key_values,
                layer_idx=self.layer_idx,
                padding_mask=padding_mask,
            )

            if is_sparse:
                # CSA: compressed_kv 是 per-query 的 (batch_size, q_len, topk, head_dim)
                # 不直接与 window kv 沿着 kv 轴拼接
                o = sparse_csa_attention(
                    q=q,
                    window_kv=kv,
                    compressed_kv=compressed_kv,
                    window_mask=attention_mask,
                    compressed_bias=block_bias,
                    compressed_index_bias=(
                        compressed_index_bias * self.index_score_bias_alpha
                        if compressed_index_bias is not None and self.index_score_bias_alpha != 0.0
                        else None
                    ),
                    attn_sink=self.attn_sink,
                    softmax_scale=self.scaling,
                )
            else:
                # HCA: compressed_kv 是 dense 的 (batch_size, 1, n_blocks, head_dim)
                # 可以直接与 window kv 沿着 kv 轴拼接
                kv = torch.cat([kv, compressed_kv], dim=2)  # 在 kv_len 维度上拼接 (batch_size, 1, kv_len + n_blocks, head_dim)
                if attention_mask is not None:
                    # (batch_size, 1, q_len, kv_len) -> (batch_size, 1, q_len, kv_len + n_blocks)
                    attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
                o = dense_shared_kv_attention(
                    q=q,
                    kv=kv,
                    attention_mask=attention_mask,
                    attn_sink=self.attn_sink,
                    softmax_scale=self.scaling,
                )
        
        # 对输出应用反向的部分 RoPE
        o_nope = o[..., :-self.rope_head_dim]
        o_rope = o[..., -self.rope_head_dim:]  # (batch_size, num_attention_heads, seq_len, rope_head_dim)
        # 实数形式可通过 sin 取负来实现反向旋转
        inverse_position_embeddings = (position_embeddings[0], -position_embeddings[1])
        o_rope = apply_rotary_emb(o_rope, inverse_position_embeddings)
        o = torch.cat([o_nope, o_rope], dim=-1)  # 重新拼接 (batch_size, num_attention_heads, seq_len, head_dim)
        o = o.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_attention_heads * self.head_dim)  # (batch_size, seq_len, num_attention_heads * head_dim)

        # 分组投影输出
        o = o.view(batch_size, seq_len, self.o_groups, -1)  # (batch_size, seq_len, o_groups, num_attention_heads * head_dim // o_groups)
        wo_a = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)  # (o_groups, o_lora_rank, num_attention_heads * head_dim // o_groups)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)  # (batch_size, seq_len, o_groups, o_lora_rank)
        hidden_states = self.wo_b(o.flatten(2))  # (batch_size, seq_len, hidden_size)
        return hidden_states
