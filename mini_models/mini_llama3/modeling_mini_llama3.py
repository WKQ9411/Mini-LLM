import torch
from torch import nn
from typing import Union, Optional
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin

from ..base_module import RMSNorm, SwiGLUFFN
from ..attention import StandardAttention
from ..rope import RotaryEmbedding
from .configuration_mini_llama3 import MiniLlama3Config

# 参考代码:
# - llama 官方仓库源码: https://github.com/meta-llama/llama3/blob/main/llama/model.py
# - transformers 源码: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py


# mini_llama3 解码器层
class MiniLlama3DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, config: MiniLlama3Config):
        super().__init__()

        # 定义参数
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx

        # self-attention
        self.self_attn = StandardAttention(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            attention_bias=config.attention_bias,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # feedforward
        self.mlp = SwiGLUFFN(dim=config.hidden_size, inter_dim=config.intermediate_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # feedforward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# mini_llama3 抽象基类
class MiniLlama3PreTrainedModel(PreTrainedModel):
    config: MiniLlama3Config  # 用于类型标注(type hint)
    base_model_prefix = "model"  # 定义模型主干模块的属性名
    config_class = MiniLlama3Config  # 用于 transformers 框架的模型注册机制，类属性(class level)


# mini_llama3 主干模型
class MiniLlama3Model(MiniLlama3PreTrainedModel):
    def __init__(self, config: MiniLlama3Config):
        super().__init__(config)  # 调用父类初始化方法，会有 self.config = config，实例属性(instance level)
        self.vocab_size = config.vocab_size
        self.padding_idx = getattr(config, "pad_token_id", None)

        # 词嵌入层
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,  # pad token 不应参与嵌入计算
        )

        # 多层解码器
        self.layers = nn.ModuleList(
            [
                MiniLlama3DecoderLayer(layer_idx, config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # 输出归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 旋转位置编码层
        self.rotary_emb = RotaryEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
        )

        # 调用父类方法，其中主要会进行：
        #   - init_weights 初始化权重
        #   - tie_weights 将输入 Embedding 和 输出 lm_head 进行权重共享，使其语义空间一致，此外还能减小参数量
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        前向传播

        Args:
            input_ids: 输入 token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, 1, q_len, kv_len) 或 (batch_size, number_of_seen_tokens + q_len)
            position_ids: 位置索引 (batch_size, seq_len)
            past_key_values: 继承自 Cache 基类, 用于缓存和管理 KV Cache
            inputs_embeds: 嵌入向量 (batch_size, seq_len, hidden_size)
            cache_position: 缓存位置索引 (seq_len,) 或 (batch_size, seq_len)
            use_cache: 是否使用缓存

        Returns:
            BaseModelOutputWithPast: 包含 hidden_states 和 past_key_values 的输出
        """
        # ^ 是异或运算符，只有一个是 True 时为 True，即 input_ids 和 inputs_embeds 只能提供一个
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 获取嵌入向量
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        # DynamicCache 用于动态缓存 KV 值，StaticCache 则是静态预留
        # TODO: 当前直接使用 Cache 类，后续可以进行一些兼容的自定义实现
        # 参考 DynamicCache 的注释:
        # A cache that grows dynamically as more tokens are generated. This is the default for generative models.
        # It stores the key and value states as a list of `CacheLayer`, one for each layer.
        # The expected shape for each tensor in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # cache_position 是当前输入序列的位置索引，索引范围为 [past_seen_tokens, past_seen_tokens + seq_len]
        # 形状为 (seq_len,)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # position_ids 同样是位置索引，形状为 (batch_size, seq_len)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        # 创建因果掩码
        # TODO: 当前直接使用 transformers 的实现，后续可以进行一些兼容的自定义实现
        # attention_mask 的含义参考 create_causal_mask 的注释:
        # The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens + q_length).
        # It can also be an already prepared 4D mask, in which case it is returned as-is.
        # 即默认情况下是一个 2D 的 pad mask，形状为 (batch_size, number_of_seen_tokens + q_len)，通常就是 (batch_size, kv_len)
        # 如果已经是一个准备好的 4D mask，则直接原样返回，形状是 (batch_size, 1, q_len, kv_len)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # 解码器逐层前向传播
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # 输出归一化
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# mini_llama3 因果语言模型
class MiniLlama3ForCausalLM(MiniLlama3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]  # 声明需要共享的权重
    architecture_type = "Dense"  # 自定义字段

    def __init__(self, config: MiniLlama3Config):
        super().__init__(config)
        self.model = MiniLlama3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        前向传播

        Args:
            input_ids: 输入 token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, 1, q_len, kv_len) 或 (batch_size, number_of_seen_tokens + q_len)
            position_ids: 位置索引 (batch_size, seq_len)
            past_key_values: 继承自 Cache 基类, 用于缓存和管理 KV Cache
            inputs_embeds: 输入嵌入向量 (batch_size, seq_len, hidden_size)
            labels: 用于计算损失的目标 token ids (batch_size, seq_len)
            use_cache: 是否使用缓存
            cache_position: 缓存位置索引 (seq_len,) 或 (batch_size, seq_len)
            logits_to_keep: 保留的 logits 数量，默认为 0 表示不进行任何过滤

        Returns:
            CausalLMOutputWithPast: 包含 logits 和损失的字典
        """
        # 主干模型前向传播
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # 关于 logits_to_keep，transformers 中的描述如下：
        #   If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
        #   `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
        #   token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
        #   If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
        #   This is useful when using packed tensor format (single dimension for batch and sequence length).
        # 因此训练时应计算全部的 logits，即 logits_to_keep = 0，推理时一般只取最后一个 logits，即 logits_to_keep = 1
        # slice(-logits_to_keep, None) 等价于 -logits_to_keep:
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:  # 训练阶段
            # transformers 的 loss_function 会在内部对 label 进行 shift 操作
            # 需注意这里的 labels 是还未进行 shift 的，实际上就是 input_ids 本身
            # 详见 transformer.loss.loss_utils.py 的 ForCausalLMLoss
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
