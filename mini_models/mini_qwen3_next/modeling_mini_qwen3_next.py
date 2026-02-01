import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Optional
from dataclasses import dataclass
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin

from ..base_module import ZeroCenteredRMSNorm, SwiGLUFFN
from ..attention import GatedDeltaNet, GatedAttention
from ..rope import RotaryEmbedding
from ..cache import MiniQwen3NextDynamicCache
from .configuration_mini_qwen3_next import MiniQwen3NextConfig

# 参考代码:
# - transformers 源码: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py


# 前馈网络
class MiniQwen3NextMLP(SwiGLUFFN):
    pass


# mini_qwen3_next 自定义输出类
@dataclass
class MiniQwen3NextModelOutput(BaseModelOutputWithPast):
    """自定义输出类，包含额外的字段"""
    router_logits: Optional[list[dict]] = None


# mini_qwen3_next 自定义输出类
@dataclass
class MiniQwen3NextForCausalLMOutput(CausalLMOutputWithPast):
    """自定义输出类，包含额外的字段"""
    aux_loss: Optional[torch.Tensor] = None
    router_logits: Optional[list[dict]] = None
    all_global_counts: Optional[list[dict]] = None


# mini_qwen3_next 路由专家网络
class MiniQwen3NextExperts(nn.Module):
    def __init__(self, config: MiniQwen3NextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.moe_intermediate_size, self.hidden_size))  # 这里相当于合并了 gate_proj 和 up_proj
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.moe_intermediate_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        专家网络前向传播
        
        Args:
            hidden_states (torch.Tensor): 输入张量 (batch_size * seq_len, hidden_size)
            top_k_index (torch.Tensor): 选中的专家索引 (batch_size * seq_len, top_k)
            top_k_weights (torch.Tensor): 选中的专家权重 (batch_size * seq_len, top_k)
        
        Returns:
            torch.Tensor: 输出张量 (batch_size * seq_len, hidden_size)
        """
        final_hidden_states = torch.zeros_like(hidden_states)  # 初始化输出张量 (batch_size * seq_len, hidden_size)
        
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)  # 将选中专家索引转换为 onehot 向量 (batch_size * seq_len, top_k, num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, batch_size * seq_len)
            # expert_mask.sum(dim=(-1, -2)) 沿着后两个维度求和，得到每个专家被选中的次数 (num_experts,)
            # torch.greater(..., 0) 比较是否大于 0，返回布尔张量，True 表示该专家被选中至少一次 (num_experts,)
            # .nonzero() 获取布尔张量的非零索引 (num_chosen_experts, 1)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:  # epxpert_idx 是一个形状为(1,) 的张量
            expert_idx = expert_idx[0]  # 取出标量值
            
            # torch.where(tensor) 查找所有非零元素的位置
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])  # 获取选中该专家的 token 索引以及该选中是 top 几
            current_state = hidden_states[token_idx]  # 获取选中该专家的所有 token (num_selected_tokens, hidden_size)
            
            # 对选中的专家进行前向计算，相当于 SwiGLU
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)  # gate 和 up 形状均为 (num_selected_tokens, moe_intermediate_size)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])  # (num_selected_tokens, hidden_size)
            
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]  # 为当前专家的输出乘以对应的 top_k 权重
            # 原地累加操作，将 current_hidden_states 的值按照 token_idx 索引加到 final_hidden_states 上
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))  # (batch_size * seq_len, hidden_size)

        return final_hidden_states


# mini_qwen3_next 专家路由层
class MiniQwen3NextTopKRouter(nn.Module):
    def __init__(self, config: MiniQwen3NextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # 确保是逐 token 视角 (batch_size * seq_len, hidden_size)
        
        # F.linear 相当于 hidden_states @ weight.T
        router_logits = F.linear(hidden_states, self.weight)  # 每个专家的得分 logits (batch_size * seq_len, num_experts)
        # 源代码这里 softmax 后仍然使用变量名 router_logits，但后期计算负载均衡辅助损失时，会再使用一次 softmax，导致分布进一步变尖锐
        # 因此这里我重命名为 router_softmax，router_logits 保持原始的 logits 含义
        router_softmax = F.softmax(router_logits, dtype=torch.float, dim=-1)  # 每个专家的概率，DeepSeek V3 中使用的是 sigmoid
        
        # 选出 top_k 个概率最大的专家得分和索引
        router_top_value, router_indices = torch.topk(router_softmax, self.top_k, dim=-1)  # (batch_size * seq_len, top_k)
        if self.norm_topk_prob:  # 对选中的专家得分进行归一化
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        
        router_scores = router_top_value.to(router_softmax.dtype)
        
        return router_logits, router_scores, router_indices


# mini_qwen3_next MoE 模块
class MiniQwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, config: MiniQwen3NextConfig):
        super().__init__()
        self.gate = MiniQwen3NextTopKRouter(config)
        self.experts = MiniQwen3NextExperts(config)
        self.shared_expert = MiniQwen3NextMLP(config.hidden_size, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)  # 共享专家门控

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_size)  # 逐 token 视角 (batch_size * seq_len, hidden_size)
        
        shared_expert_output = self.shared_expert(hidden_states_reshaped)  # 共享专家输出 (batch_size * seq_len, hidden_size)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)  # 获取路由权重和选中的专家索引
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)  # 路由专家输出 (batch_size * seq_len, hidden_size)

        # 这里对共享专家的输出进行了 token 级的门控，与 DeepSeek V3 略有不同
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        # 最终专家输出 (batch_size, seq_len, hidden_size)
        expert_output += shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_size)
        
        return expert_output, router_logits


# mini_qwen3_next 解码器层
class MiniQwen3NextDecoderLayer(nn.Module):
    def __init__(self, config: MiniQwen3NextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # attention
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                layer_norm_epsilon=config.rms_norm_eps,
                )
        elif self.layer_type == "full_attention":
            self.self_attn = GatedAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                rope_theta=config.rope_theta,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                )

        # feedforward
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):  # decoder_sparse_step 用于指定每隔多少层放置一个 MoE 层，设置为 1 表示每层都是 MoE
            self.mlp = MiniQwen3NextSparseMoeBlock(config)
        else:
            self.mlp = MiniQwen3NextMLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MiniQwen3NextDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # attention
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # feedforward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        output = self.mlp(hidden_states)
        router_logits = None
        if isinstance(output, tuple):  # 解包 MoE 输出的 router_logits
            hidden_states, router_logits = output
        else:
            hidden_states = output
        hidden_states = residual + hidden_states

        return hidden_states, router_logits
    

# mini_qwen3_next 抽象基类
class MiniQwen3NextPreTrainedModel(PreTrainedModel):
    config: MiniQwen3NextConfig  # 用于类型标注(type hint)
    base_model_prefix = "model"  # 定义模型主干模块的属性名
    config_class = MiniQwen3NextConfig  # 用于 transformers 框架的模型注册机制，类属性(class level)
    
    @torch.no_grad()
    def _init_weights(self, module):
        # 对部分模块所定义的参数设定初始化方式，参考 PreTrainedModel 类的 _init_weights 方法注释
        # Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        # initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        # `nn.Parameter`, this method should also be overridden in order to initialize it correctly
        # 默认的 _init_weights 方法会初始化大部分常见模块，如 nn.Linear, nn.Embedding, nn.LayerNorm 等
        super()._init_weights(module)  # 调用 PreTrainedModel 类的初始化方法
        if isinstance(module, GatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.uniform_(0, 16).log_()
        elif isinstance(module, ZeroCenteredRMSNorm):
            module.weight.data.zero_()
        elif isinstance(module, MiniQwen3NextExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.down_proj.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, MiniQwen3NextSparseMoeBlock):
            module.gate.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


# mini_qwen3_next 主干模型
class MiniQwen3NextModel(MiniQwen3NextPreTrainedModel):
    def __init__(self, config: MiniQwen3NextConfig):
        super().__init__(config)  # 调用父类初始化方法，会有 self.config = config，实例属性(instance level)
        self.padding_idx = getattr(config, "pad_token_id", None)
        
        # 词嵌入层
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,  # pad token 不应参与嵌入计算
        )
        
        # 多层解码器
        self.layers = nn.ModuleList(
            [MiniQwen3NextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # 输出归一化层
        self.norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
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
        attention_mask: Optional[torch.Tensor] | None = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MiniQwen3NextDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> MiniQwen3NextModelOutput:
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
            MoeModelOutputWithPast: 包含 hidden_states 和 past_key_values 的输出
        """
        # ^ 是异或运算符，只有一个是 True 时为 True，即 input_ids 和 inputs_embeds 只能提供一个
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 获取嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 初始化 Cache
        if use_cache and past_key_values is None:
            past_key_values = MiniQwen3NextDynamicCache(config=self.config)

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
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 解码器逐层前向传播
        all_router_logits = []
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            hidden_states, router_logits = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if router_logits is not None:
                all_router_logits.append({"layer_idx": decoder_layer.layer_idx, "router_logits": router_logits})

        # 输出归一化
        hidden_states = self.norm(hidden_states)

        return MiniQwen3NextModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        以下情况无需使用 mask
            1. 已经有 Cache 的前向, 直接谁用 Cache 的最后状态
            2. 预训练关注序列中的所有 token
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask


# 负载均衡损失
def load_balancing_loss_func(
    gate_logits: list[dict] | None,
    num_experts: int | None = None,
    top_k: int | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    """
    计算负载均衡损失, 参考 Switch Transformer (https://huggingface.co/papers/2101.03961) 中的公式 (4)-(6)

    Args:
        gate_logits: 所有专家层的门控 logits, 每层的形状是 (batch_size * seq_len, num_experts)
        num_experts: 专家数量
        top_k: 选中的专家数量
        attention_mask: 注意力掩码 (batch_size, seq_len)

    Returns:
        负载均衡辅助损失
    """
    if not gate_logits:
        return 0

    compute_device = gate_logits[0]["router_logits"].device
    # NOTE: transformers 中 Qwen3-Next 的负载均衡将每层的 logits 拼接成 (batch_size * seq_len * layers, num_experts) 的形状
    # 个人认为这样的计算方式会导致负载均衡是基于整个模型的，而不是每层的，可能出现对某个专家 id，它在某层激活极高，而另一层几乎不激活，但在整个模型中却具有合适的 f_i,P_i，这是不合理的
    # 因此这里我们分层计算 f_i 和 P_i，然后求和得到最终的损失
    # 实际上，DeepSeekV3 的序列级辅助损失就是对此的进一步细分，在序列上计算 f_i 和 P_i
    overall_loss = 0
    for layer_gate in gate_logits:
        layer_gate_logits = layer_gate["router_logits"].to(compute_device)  # (batch_size * seq_len, num_experts)
        # 计算每个专家的选中概率 (batch_size * seq_len, num_experts)
        routing_weights = F.softmax(layer_gate_logits, dtype=torch.float, dim=-1)
        # 获取选中的专家索引 (batch_size * seq_len, top_k)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        # 转换为 one-hot 向量 (batch_size * seq_len, top_k, num_experts)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts)

        if attention_mask is None:  # 所有 token 选的专家都进行计算
            # 计算每个专家作为 top 几的平均被选中次数 (top_k, num_experts) 即 f_i
            # 若果再沿 top_k 维度求和，则是每个专家总的平均被选中次数，在最后的 torch.sum 中实现了这种求和
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
            # 计算每个专家在所有 token 上的平均概率 (num_experts,) 即 P_i
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:  # 仅计算关注序列的 token 选的专家
            batch_size, sequence_length = attention_mask.shape

            # 计算与 expert_mask 形状相同的 mask (batch_size * seq_len, top_k, num_experts)
            expert_attention_mask = (
                attention_mask[:, :, None, None]
                .expand((batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
            )
            # 计算 f_i，只计算有效 token
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

            # 计算与 routing_weights 形状相同的 mask (batch_size * seq_len, num_experts)
            router_per_expert_attention_mask = (
                attention_mask[:, :, None]
                .expand((batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
            )
            # 计算 P_i，只计算有效 token
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(router_per_expert_attention_mask, dim=0)

        # 相当于完成了 top_k 方向的求和和 i=1 到 N 的求和
        # 理想情况下，每个专家被选中的频率为 top_k / num_experts，每个专家被选中的概率为 1 / num_experts
        # 所有专家的 f_i 和 P_i 的乘积之和为 ∑(f_i * P_i) = top_k / num_experts
        # 乘以 num_experts / top_k 以统一尺度后，每层的理想 loss 则为 1
        loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        loss = loss * num_experts / top_k  # 乘 N 以统一尺度
        overall_loss += loss

    return overall_loss / len(gate_logits)  # 平均层数，将最终 aux_loss 归一化为 1，方便在不同设置下比较


# mini_qwen3_next 因果语言模型
class MiniQwen3NextForCausalLM(MiniQwen3NextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    architecture_type = "Linear"  # 自定义字段

    def __init__(self, config: MiniQwen3NextConfig):
        super().__init__(config)
        self.model = MiniQwen3NextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MiniQwen3NextDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> MiniQwen3NextForCausalLMOutput:
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
            output_router_logits: 是否输出路由 logits
            cache_position: 缓存位置索引 (seq_len,) 或 (batch_size, seq_len)
            logits_to_keep: 保留的 logits 数量，默认为 0 表示不进行任何过滤

        Returns:
            CausalLMOutputWithPast: 包含 logits 和损失的字典
        """
        # 配置是否输出路由 logits
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits

        # 主干模型前向传播
        outputs: MiniQwen3NextModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
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
                vocab_size=self.vocab_size,
                **kwargs,
            )

        # 负载均衡损失
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        # 计算每层专家负载情况
        # 与 mini_deepseekv3 不同，这里简化一下，不再计算全部 GPU 的平均负载情况，减小通信开销
        all_global_counts = None
        if output_router_logits and outputs.router_logits:
            all_global_counts = []
            for layer_gate in outputs.router_logits:
                routing_weights = F.softmax(layer_gate["router_logits"], dtype=torch.float, dim=-1)
                _, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
                counts = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts)
                all_global_counts.append({"layer_idx": layer_gate["layer_idx"], "global_counts": counts.cpu().tolist()})

        return MiniQwen3NextForCausalLMOutput(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            all_global_counts=all_global_counts,
        )
