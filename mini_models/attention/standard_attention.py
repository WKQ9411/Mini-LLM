from typing import Tuple, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from .utils import repeat_kv
from ..rope import apply_rotary_emb


class StandardAttention(nn.Module):
    """
    标准注意力

    Args:
        layer_idx (int): 层索引
        hidden_size (int): 隐状态维度
        num_attention_heads (int): 注意力头数, 即 query 头数
        max_position_embeddings (int): 最大位置编码长度
        rope_theta (float): RoPE 的底数, 默认为 10000.0
        num_key_value_heads (Optional[int]): key-value 头数, 如果为 None, 则与 query 头数相同, 此时为 MHA
        head_dim (Optional[int]): 每个头的维度, 如果为 None, 则使用 hidden_size // num_attention_heads
        attention_bias (bool): 是否使用注意力偏置, 默认为 False
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000.0,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = head_dim

        # 计算重复次数：每个 kv 头对应的 query 头数
        assert self.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # 线性变换层
        self.q_proj = nn.Linear(hidden_size, self.num_attention_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)

        # 注意力缩放因子
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        兼容 transformers 的 attention 前向传播

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
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)    # (batch_size, n_kv_heads, seq_len, head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (batch_size, n_kv_heads, seq_len, head_dim)

        # step 2. 应用 RoPE, 其中 position_embeddings 已进行了位置对齐
        # 由于 RoPE 可注入相对位置信息，每个 attention 层都可使用 RoPE，避免位置信息在深层网络稀释
        cos, sin = position_embeddings
        query_states = apply_rotary_emb(query_states, position_embeddings)
        key_states = apply_rotary_emb(key_states, position_embeddings)

        # step 3. 更新缓存，训练阶段 past_key_values 为 None
        # kv cache 管理有大量可以优化的地方，因此 transformers 将其封装为了 Cache 类，便于各种自定义改进和优化
        # 这里先使用 transformers 提供的接口，后续可以进行一些兼容的自定义实现
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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

        # 投影输出
        attn_output = self.o_proj(attn_output)  # (batch_size, seq_len, hidden_size)

        return attn_output, attn_weights