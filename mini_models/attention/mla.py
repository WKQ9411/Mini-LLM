from typing import Tuple, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from ..rope import apply_rotary_emb
from ..base_module import RMSNorm


class MultiHeadLatentAttention(nn.Module):
    """
    多头潜在注意力, Multi-Head Latent Attention (MLA), DeepSeekV3 版本

    Args:
        layer_idx (int): 层索引
        hidden_size (int): 隐状态维度【对应论文中的 d】
        num_attention_heads (int): 注意力头数【对应论文中的 n_h】
        q_lora_rank (int): query 的下投影维度【对应论文中的 d_c'】
        kv_lora_rank (int): key/value 的下投影维度【对应论文中的 d_c】
        qk_nope_head_dim (int): 没有位置编码的 q_t^C 和 k_t^C 的每个头的维度【对应论文中的 d_h】
        qk_rope_head_dim (int): 解耦的带 RoPE 的 q_t^R 和 k_t^R 的每个头的维度【对应论文中的 d_h^R】
        qk_head_dim (int): 最终执行注意力计算的 query 和 key 的每个头的维度【即 d_h + d_h^R】
        v_head_dim (int): value 的每个头的维度, 可以与 qk_nope_head_dim 不同【但在论文中也同样设定为 d_h】
        num_key_value_heads (Optional[int]): key-value 头数, 如果为 None, 则与 query 头数相同
        attention_bias (bool): 是否使用注意力偏置, 默认为 False
        attn_impl (str): 注意力实现方式, 可选 "naive" 或 "absorb", 即原始方式或矩阵吸收方式, 默认为 "absorb"
    """
    def __init__(
        self, 
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        num_key_value_heads: Optional[int] = None,
        attention_bias: bool = False,
        attn_impl: str = "absorb"
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_impl = attn_impl
        
        # DeepSeekV3 的 MLA 实现中，使用的均为 n_heads，因此以下逻辑实际暂时用不到
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        

        # 定义 query/key/value 维度
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # 低秩压缩 query
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=attention_bias)  # 下投影: d -> d_c'
        self.q_norm = RMSNorm(self.q_lora_rank)  # 下投影后对潜在向量进行一次 RMSNorm，原文中似乎没提到，但源码中有
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_attention_heads * self.qk_head_dim, bias=attention_bias)  # 同时进行上投影 + 解耦多头 query 投影: d_c' -> (d_h + d_h^R) * n_h
        
        # key / value 的维度变换
        self.wkv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=attention_bias)  # 同时进行下投影 + 解耦共享 key 投影: d -> d_c + d_h^R
        self.kv_norm = RMSNorm(self.kv_lora_rank)  # 对潜在向量进行 RMSNorm
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=attention_bias)  # 同时进行 key 和 value 的上投影: d_c -> (d_h + d_h) * n_h

        # 输出的维度变换: d_h * n_h -> d
        self.wo = nn.Linear(self.num_attention_heads * self.v_head_dim, self.hidden_size, bias=attention_bias)

        self.scaling = self.qk_head_dim ** -0.5  # 注意力缩放因子，即 1/sqrt(d_h + d_h^R)  
    
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
        MLA 的前向传播

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, dim)
            position_embeddings (Optional[tuple[Tensor, Tensor]]): 预计算 (cos, sin) 表, 形状 (batch_size, seq_len, head_dim)
            attention_mask (Optional[torch.Tensor]): 通常为 (batch, 1, q_len, kv_len) 的加性掩码
            past_key_values (Optional[Cache]): transformers 缓存对象, 此处仅做占位兼容
            cache_position (Optional[LongTensor]): 当前位置索引 (q_len,) 或 (batch, q_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输出张量 (attn_output, attn_weights)
        """
        batch_size, seq_length = hidden_states.shape[:-1]  # (batch_size, seq_len)
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)  # (batch_size, seq_len, n_heads, qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)  # (batch_size, seq_len, n_heads, qk_nope_head_dim + v_head_dim)
        cos, sin = position_embeddings

        # -------------------------- query 部分 --------------------------
        # 同步计算 q_t^C 和 q_t^R
        q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))  # (batch_size, seq_len, n_heads * qk_head_dim)
        q = q.view(query_shape).transpose(1, 2)  # 划分多头: (batch_size, n_heads, seq_len, qk_head_dim)
        # 将 q 拆分成不带位置编码的 q_nope 部分和带位置编码的 q_pe 部分
        # q_nope: (batch_size, n_heads, seq_len, qk_nope_head_dim)
        # q_pe: (batch_size, n_heads, seq_len, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, position_embeddings)  # 对解耦部分应用旋转位置编码

        # ----------------------- key / value 部分 -----------------------
        # 同步计算 k_t^C 和 k_t^R
        kv = self.wkv_a(hidden_states)  # 同时进行低秩压缩和解耦 key 的变换 (batch_size, seq_len, kv_lora_rank + qk_rope_head_dim)
        # 将上述结果拆分成潜在向量 kv 部分和带位置编码的 k_pe 部分
        # kv: (batch_size, seq_len, kv_lora_rank)
        # k_pe: (batch_size, seq_len, qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(1), position_embeddings)  # 先增加 head 维度 (batch_size, 1, seq_len, qk_rope_head_dim)，而后对解耦部分应用旋转位置编码
        
        # --------------------------- 注意力计算 ---------------------------
        # 注意力实现分为两种方式：原始方式和矩阵吸收方式
        # 原始方式 (naive)：从潜在向量中恢复出 key/value 后，执行原始注意力计算，这也是 transformers 中的 DeepSeekV3 实现的方式, 缓存的是恢复后的 key/value
        if self.attn_impl == "naive":
            kv = self.wkv_b(self.kv_norm(kv))  # 上投影: (batch_size, seq_len, n_heads * (qk_nope_head_dim + v_head_dim))
            kv = kv.view(key_shape).transpose(1, 2)  # 划分多头: (batch_size, n_heads, seq_len, qk_nope_head_dim + v_head_dim)
            # 将 kv 拆分成不带位置编码的 k_nope 部分和 value_states 部分
            # k_nope: (batch_size, n_heads, seq_len, qk_nope_head_dim)
            # value_states: (batch_size, n_heads, seq_len, v_head_dim)
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # 将 k_pe 扩展到与 n_heads 匹配的维度，因为 k_pe 对所有头共享
            k_pe = k_pe.expand(batch_size, self.num_attention_heads, seq_length, self.qk_rope_head_dim)  # (batch_size, n_heads, seq_len, qk_rope_head_dim)

            # 执行注意力计算的 query 和 key
            query_states = torch.cat([q_nope, q_pe], dim=-1)  # (batch_size, n_heads, seq_len, qk_head_dim)
            key_states = torch.cat([k_nope, k_pe], dim=-1)  # (batch_size, n_heads, seq_len, qk_head_dim)

            # 更新缓存，训练阶段 past_key_values 为 None
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
            # 计算缩放点积注意力，因为均采用 n_heads，因此无需 repeat_kv
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling  # (batch_size, n_heads, q_len, k_len)
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # Softmax 操作对数值稳定性要求较高，因此在 float32 下进行计算以避免溢出或下溢，然后再转换为原数据类型
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # 计算输出
            attn_output = torch.matmul(attn_weights, value_states)  # (batch_size, n_heads, q_len, v_head_dim)

            # 确保输出张量是连续的
            attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, q_len, n_heads, v_head_dim)
            attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()  # (batch_size, seq_len, n_heads * v_head_dim)
        
        # 矩阵吸收方式 (absorb)：使用矩阵吸收的方式，将潜在向量吸收到权重矩阵中，从而避免恢复出 key/value 的过程, 缓存的是压缩后的潜在向量和解耦的携带位置编码信息的 k_pe
        elif self.attn_impl == "absorb":
            # transformers 中的 DeepSeekV3 实现没有使用矩阵吸收，且缓存的不是压缩后的潜在向量，而是上投影后的 key/value
            # 这里我们缓存压缩后的潜在向量和解耦的携带位置编码信息的 k_pe
            wkv_b = self.wkv_b.weight  # weight 的形状为(out_features, in_features), 即(n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
            wkv_b = wkv_b.view(self.num_attention_heads, -1, self.kv_lora_rank)  # (n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
            # q_nope: (batch_size, n_heads, seq_len, qk_nope_head_dim)
            # wkv_b 截取上投影恢复 key 的权重: (n_heads, qk_nope_head_dim, kv_lora_rank), 即每个头的权重形状是(qk_nope_head_dim, kv_lora_rank), 该权重由 q_nope 吸收
            q_nope = torch.einsum("bhsd,hdc->bhsc", q_nope, wkv_b[:, :self.qk_nope_head_dim])  # (batch_size, n_heads, seq_len, kv_lora_rank)

            # 由于使用了矩阵吸收，可以将潜在向量和携带位置编码信息的 k_pe 直接缓存，这里直接将二者传入给 DynamicCache 中
            kv = self.kv_norm(kv).unsqueeze(1)  # (batch_size, 1, seq_len, kv_lora_rank) 增加 head 维度后缓存
            if past_key_values is not None:
                # DynamicCache 通常是模型默认 Cache 类，无需传入 cache_kwargs，"naive" 中传入的 cache_kwargs 对于 DynamicCache 实际上也没什用
                # 因此这里索性直接将 kv 传入给了 DynamicCache 的 key_states，而 k_pe 传入给了 DynamicCache 的 value_states
                kv, k_pe = past_key_values.update(kv, k_pe, self.layer_idx)
            
            # 计算注意力得分，nope 部分与 rope 部分分别计算后相加
            # 通过解耦 nope 和 rope，从而能够对 nope 部分进行矩阵吸收，使得缓存潜在向量 kv 成为可能，减少了缓存压力，计算注意力得分时，只要两部分分别计算并相加即可
            attn_weights = (torch.matmul(q_nope, kv.transpose(2, 3)) + torch.matmul(q_pe, k_pe.transpose(2, 3))) * self.scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : kv.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # Softmax 操作对数值稳定性要求较高，因此在 float32 下进行计算以避免溢出或下溢，然后再转换为原数据类型
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_nope.dtype)  # (batch_size, n_heads, q_len, kv_len)

            # 计算输出，首先直接使用 attn_weights 聚合缓存的潜在向量 kv 
            # (batch_size, n_heads, q_len, kv_len) (batch_size, 1, kv_len, kv_lora_rank) -> (batch_size, n_heads, q_len, kv_lora_rank)
            attn_output = torch.matmul(attn_weights, kv)
            # 然后吸收 value 的上投影矩阵
            attn_output = torch.einsum("bhsc,hdc->bhsd", attn_output, wkv_b[:, self.qk_nope_head_dim:])  # (batch_size, n_heads, q_len, v_head_dim)

            # 确保输出张量是连续的
            attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, q_len, n_heads, v_head_dim)
            attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()  # (batch_size, q_len, n_heads * v_head_dim)

        else:
            raise ValueError(f"Invalid attention implementation: {self.attn_impl}, must be 'naive' or 'absorb'")
        
        # 投影输出
        attn_output = self.wo(attn_output)  # (batch_size, seq_len, hidden_size)

        return attn_output, attn_weights