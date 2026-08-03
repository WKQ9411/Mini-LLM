import torch
from torch import nn

from mini_models.mini_llama3.configuration_mini_llama3 import MiniLlama3Config

from mini_inference.layers.activation import SiluAndMul
from mini_inference.layers.attention import Attention
from mini_inference.layers.layernorm import RMSNorm
from mini_inference.layers.linear import MergedLinear
from mini_inference.layers.rotary_embedding import get_rope
from mini_inference.utils.context import get_context


class MiniLlama3Attention(nn.Module):

    def __init__(self, config: MiniLlama3Config, attention_backend: str) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = MergedLinear(
            config.hidden_size,
            [self.q_size, self.kv_size, self.kv_size],
            bias=config.attention_bias,
        )  # 合并投影
        self.o_proj = nn.Linear(
            self.q_size,
            config.hidden_size,
            bias=config.attention_bias,
        )

        rope_parameters = config.rope_parameters or {}
        rope_type = rope_parameters.get("rope_type", "default")
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_parameters.get("rope_theta", 10000.0),
            rope_type=rope_type,
            factor=rope_parameters.get("factor"),
            attention_factor=rope_parameters.get("attention_factor"),
            beta_fast=rope_parameters.get("beta_fast", 32),
            beta_slow=rope_parameters.get("beta_slow", 1),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            attention_backend,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)  # (num_tokens, num_heads, head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)  # (num_tokens, num_kv_heads, head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)  # (num_tokens, num_kv_heads, head_dim)
        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)  # (num_tokens, num_heads, head_dim)
        return self.o_proj(output.flatten(1, -1))  # (num_tokens, hidden_size)


class MiniLlama3MLP(nn.Module):

    def __init__(self, config: MiniLlama3Config) -> None:
        super().__init__()
        self.gate_up_proj = MergedLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.w2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return self.w2(hidden_states)


class MiniLlama3DecoderLayer(nn.Module):

    def __init__(self, config: MiniLlama3Config, attention_backend: str) -> None:
        super().__init__()
        self.self_attn = MiniLlama3Attention(config, attention_backend)
        self.mlp = MiniLlama3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiniLlama3Model(nn.Module):

    def __init__(self, config: MiniLlama3Config, attention_backend: str) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            MiniLlama3DecoderLayer(config, attention_backend)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MiniLlama3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "w1": ("gate_up_proj", 0),
        "w3": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: MiniLlama3Config,
        attention_backend: str = "triton",
    ) -> None:
        super().__init__()
        self.model = MiniLlama3Model(config, attention_backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        return self.lm_head(hidden_states)
