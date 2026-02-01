from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache
from .utils import repeat_kv
from ..rope import apply_rotary_emb
from ..cache import MiniQwen3NextDynamicCache
from ..base_module import ZeroCenteredRMSNorm


class GatedAttention(nn.Module):
    """
    带门控的注意力机制, 与标准注意力类似, 但额外加入 qk norm 和 输出门控

    Args:
        layer_idx (int): 层索引
        hidden_size (int): 隐状态维度
        num_attention_heads (int): 注意力头数, 即 query 头数
        rope_theta (float): RoPE 的底数, 默认为 10000.0
        num_key_value_heads (Optional[int]): key-value 头数, 如果为 None, 则与 query 头数相同, 此时为 MHA
        head_dim (Optional[int]): 每个头的维度, 如果为 None, 则使用 hidden_size // num_attention_heads
        attention_bias (bool): 是否使用注意力偏置, 默认为 False
        rms_norm_eps (float): qk norm 的 RMSNorm epsilon, 默认为 1e-6
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        rope_theta: float = 10000.0,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.head_dim = head_dim

        # 计算重复次数：每个 kv 头对应的 query 头数
        assert self.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # 线性变换层
        self.q_proj = nn.Linear(hidden_size, self.num_attention_heads * self.head_dim * 2, bias=attention_bias)  # 额外投影出门控张量
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)
        
        # qk norm
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=rms_norm_eps)

        # 注意力缩放因子
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[MiniQwen3NextDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Gated Attention 前向传播

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, dim)
            position_embeddings (Optional[tuple[Tensor, Tensor]]): 预计算 (cos, sin) 表, 形状 (batch_size, seq_len, head_dim)
            attention_mask (Optional[torch.Tensor]): 通常为 (batch, 1, q_len, kv_len) 的加性掩码
            past_key_values (Optional[Cache]): transformers 缓存对象, 此处仅做占位兼容
            cache_position (Optional[LongTensor]): 当前位置索引 (q_len,) 或 (batch, q_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输出张量 (attn_output, attn_weights)
        """
        input_shape = hidden_states.shape[:-1]  # (batch_size, seq_len)
        hidden_shape = (*input_shape, -1, self.head_dim)  # (batch_size, seq_len, n_(kv_)heads, head_dim)

        # step 1. 计算 query, key, value, 这里需要 hidden_states 是连续的
        query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)  # chunk 是按照所给维度均分
        gate = gate.reshape(*input_shape, -1)  # (batch_size, seq_len, n_heads * head_dim)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)              # (batch_size, n_heads, seq_len, head_dim)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # (batch_size, n_kv_heads, seq_len, head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)             # (batch_size, n_kv_heads, seq_len, head_dim)

        # step 2. 应用 RoPE
        cos, sin = position_embeddings
        query_states = apply_rotary_emb(query_states, position_embeddings)
        key_states = apply_rotary_emb(key_states, position_embeddings)

        # step 3. 更新缓存，训练阶段 past_key_values 为 None
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # 这部分参数实际没有用
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # step 4. 注意力计算
        # 重复 key 和 value 以匹配 query 头数
        key_states = repeat_kv(key_states, self.num_key_value_groups)  # (batch_size, n_heads, k_len, head_dim)
        value_states = repeat_kv(value_states, self.num_key_value_groups)  # (batch_size, n_heads, k_len, head_dim)

        # 计算缩放点积注意力
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling  # (batch_size, n_heads, q_len, k_len)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax 操作对数值稳定性要求较高，因此在 float32 下进行计算以避免溢出或下溢，然后再转换为原数据类型
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # step 5. 计算输出
        attn_output = torch.matmul(attn_weights, value_states)  # (batch_size, n_heads, q_len, head_dim)

        # 确保输出张量是连续的
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, q_len, n_heads, head_dim)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # (batch_size, seq_len, n_heads * head_dim)
        
        # 在投影前应用 gate
        attn_output = attn_output * torch.sigmoid(gate)  # 每个元素都有一个独立的门控值

        # 投影输出
        attn_output = self.o_proj(attn_output)  # (batch_size, seq_len, hidden_size)

        return attn_output, attn_weights