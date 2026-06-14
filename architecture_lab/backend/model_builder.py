import torch
from torch import nn
import torch.nn.functional as F

from mini_models.norm import RMSNorm
from mini_models.rope import RotaryEmbedding
from mini_models.ffn.swiglu import SwiGLUFFN
from mini_models.ffn.geglu import GeGLUFFN
from mini_models.ffn.moe import MoEFFN
from mini_models.attention.standard_attention import StandardAttention
from mini_models.attention.gated_attention import GatedAttention
from mini_models.attention.mla import MultiHeadLatentAttention
from mini_models.attention.gated_delta_net import GatedDeltaNet
from mini_models.attention.csa_hca import DeepSeekV4Attention


# ----------------------- 模块注册表 -----------------------
ATTENTION_REGISTRY = {
    "standard": {
        "class": StandardAttention,
        "params": {
            "num_attention_heads": {"type": "int", "default": 4, "min": 1, "max": 32},
            "num_key_value_heads": {"type": "int", "default": 2, "min": 1, "max": 32},
            "rope_theta": {"type": "float", "default": 10000.0, "min": 1, "max": 10000000},
        },
    },
    "gated": {
        "class": GatedAttention,
        "params": {
            "num_attention_heads": {"type": "int", "default": 4, "min": 1, "max": 32},
            "num_key_value_heads": {"type": "int", "default": 2, "min": 1, "max": 32},
            "rope_theta": {"type": "float", "default": 10000.0, "min": 1, "max": 10000000},
            "rms_norm_eps": {"type": "float", "default": 1e-6, "min": 1e-8, "max": 1e-2},
        },
    },
    "mla": {
        "class": MultiHeadLatentAttention,
        "params": {
            "num_attention_heads": {"type": "int", "default": 4, "min": 1, "max": 32},
            "q_lora_rank": {"type": "int", "default": 32, "min": 4, "max": 256},
            "kv_lora_rank": {"type": "int", "default": 16, "min": 4, "max": 256},
            "qk_nope_head_dim": {"type": "int", "default": 16, "min": 4, "max": 64},
            "qk_rope_head_dim": {"type": "int", "default": 8, "min": 2, "max": 32},
            "v_head_dim": {"type": "int", "default": 16, "min": 4, "max": 64},
        },
    },
    "gated_delta_net": {
        "class": GatedDeltaNet,
        "params": {
            "num_k_heads": {"type": "int", "default": 4, "min": 1, "max": 32},
            "num_v_heads": {"type": "int", "default": 4, "min": 1, "max": 32},
            "head_k_dim": {"type": "int", "default": 16, "min": 4, "max": 64},
            "head_v_dim": {"type": "int", "default": 16, "min": 4, "max": 64},
            "conv_kernel_size": {"type": "int", "default": 4, "min": 2, "max": 16},
            "layer_norm_epsilon": {"type": "float", "default": 1e-6, "min": 1e-8, "max": 1e-2},
        },
    },
    "csa_hca": {
        "class": DeepSeekV4Attention,
        "params": {
            "attention_mechanism": {"type": "select", "default": "csa", "options": ["sliding", "csa", "hca"]},
            "num_attention_heads": {"type": "int", "default": 4, "min": 1, "max": 32},
            "index_num_attention_heads": {"type": "int", "default": 2, "min": 1, "max": 16},
            "q_lora_rank": {"type": "int", "default": 32, "min": 4, "max": 256},
            "o_lora_rank": {"type": "int", "default": 16, "min": 4, "max": 128},
            "head_dim": {"type": "int", "default": 32, "min": 8, "max": 128},
            "index_head_dim": {"type": "int", "default": 16, "min": 4, "max": 64},
            "rope_head_dim": {"type": "int", "default": 8, "min": 2, "max": 64},
            "rope_theta": {"type": "float", "default": 10000.0, "min": 1, "max": 10000000},
            "o_groups": {"type": "int", "default": 2, "min": 1, "max": 8},
            "window_size": {"type": "int", "default": 32, "min": 8, "max": 256},
            "compress_ratio": {"type": "int", "default": 4, "min": 2, "max": 128},
            "index_topk": {"type": "int", "default": 2, "min": 1, "max": 16},
            "compress_rope_theta": {"type": "float", "default": 10000.0, "min": 1, "max": 10000000},
            "index_score_bias_alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
        },
    },
}

FFN_REGISTRY = {
    "swiglu": {
        "class": SwiGLUFFN,
        "params": {
            "intermediate_size": {"type": "int", "default": 256, "min": 32, "max": 2048},
        },
    },
    "geglu": {
        "class": GeGLUFFN,
        "params": {
            "intermediate_size": {"type": "int", "default": 256, "min": 32, "max": 2048},
        },
    },
    "moe": {
        "class": MoEFFN,
        "params": {
            "moe_intermediate_size": {"type": "int", "default": 256, "min": 32, "max": 2048},
            "n_routed_experts": {"type": "int", "default": 8, "min": 2, "max": 32},
            "n_shared_experts": {"type": "int", "default": 1, "min": 1, "max": 8},
            "n_activated_experts": {"type": "int", "default": 2, "min": 1, "max": 8},
            "n_expert_groups": {"type": "int", "default": 4, "min": 1, "max": 16},
            "n_limited_groups": {"type": "int", "default": 2, "min": 1, "max": 16},
            "route_scale": {"type": "float", "default": 1.0, "min": 0.1, "max": 4.0},
            "use_noaux_load_balance": {"type": "select", "default": "true", "options": ["true", "false"]},
            "use_seq_aux": {"type": "select", "default": "true", "options": ["true", "false"]},
            "seq_aux_alpha": {"type": "float", "default": 1e-4, "min": 0.0, "max": 1.0},
            "bias_update_speed": {"type": "float", "default": 1e-3, "min": 0.0, "max": 1.0},
        },
    },
}


# ----------------------- 辅助模块及函数 -----------------------
class TrainingAttentionAdapter(nn.Module):
    def __init__(self, module: nn.Module, attention_type: str):
        super().__init__()
        self.inner = module
        self.attention_type = attention_type

    def forward(self, hidden_states, position_embeddings, attention_mask, **attention_kwargs):
        # 不同模块的前向参数略有区别，这里进行适配
        if self.attention_type == "gated_delta_net":
            return self.inner(
                hidden_states=hidden_states,
                attention_mask=attention_kwargs.get("padding_mask"),   # 默认是 None
                cache_position=attention_kwargs.get("cache_position"), # 默认是 None
            )

        output = self.inner(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=attention_kwargs.get("position_ids"),
            padding_mask=attention_kwargs.get("padding_mask"),         # 默认是 None
            cache_position=attention_kwargs.get("cache_position"),     # 默认是 None
            past_key_values=attention_kwargs.get("past_key_values"),   # 默认是 None
        )
        return output[0] if isinstance(output, tuple) else output


def initialize_architecture_lab_attention(module: nn.Module, attention_type: str) -> None:
    if attention_type != "csa_hca":
        return

    # 为特定模块进行参数初始化
    if hasattr(module, "attn_sink"):
        nn.init.zeros_(module.attn_sink)
    compressor = getattr(module, "compressor", None)
    if compressor is not None and hasattr(compressor, "position_bias"):
        nn.init.zeros_(compressor.position_bias)
        indexer = getattr(compressor, "indexer", None)
        if indexer is not None and hasattr(indexer, "compressor") and hasattr(indexer.compressor, "position_bias"):
            nn.init.zeros_(indexer.compressor.position_bias)


# ----------------------- 模型构建器 -----------------------
class DecoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, ffn: nn.Module, hidden_size: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.attention = attention
        self.ffn = ffn
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, **attention_kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, position_embeddings, attention_mask, **attention_kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        aux_loss = None
        if isinstance(ffn_output, tuple):
            hidden_states, aux_loss = ffn_output
        else:
            hidden_states = ffn_output
        hidden_states = residual + hidden_states
        return hidden_states, aux_loss


class TransformerLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        max_seq_len = config["max_seq_len"]
        rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        share_embedding_head = config.get("share_embedding_head", False)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # 逐 decoder layer 构建
        self.layers = nn.ModuleList()
        for layer_idx, layer_cfg in enumerate(config["layers"]):
            attn = self._build_attention(layer_cfg["attention_type"], layer_cfg.get("attention_params", {}), hidden_size, max_seq_len, layer_idx)
            ffn = self._build_ffn(layer_cfg["ffn_type"], layer_cfg.get("ffn_params", {}), hidden_size)
            self.layers.append(DecoderLayer(attn, ffn, hidden_size, rms_norm_eps))
        
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if share_embedding_head:
            self.lm_head.weight = self.embed_tokens.weight

        # 由于 DeepSeekV4 各层可能使用不同的 rope 配置，因此为每层单独创建 rotary embedding 模块
        self.rotary_embs = nn.ModuleList()
        for layer_cfg in config["layers"]:
            rope_head_dim, rope_theta = self._get_layer_rope_config(layer_cfg)
            self.rotary_embs.append(RotaryEmbedding(max_seq_len, rope_head_dim, rope_theta=rope_theta))

        self._needs_position_ids = True

    def _resolve_attention_params(self, layer_cfg: dict) -> dict:
        cfg = ATTENTION_REGISTRY[layer_cfg["attention_type"]]
        params = {k: v["default"] for k, v in cfg["params"].items()}
        params.update(layer_cfg.get("attention_params", {}))
        return params

    def _get_layer_rope_config(self, layer_cfg: dict) -> tuple[int, float]:
        attention_type = layer_cfg["attention_type"]
        params = self._resolve_attention_params(layer_cfg)
        if attention_type in ("standard", "gated"):
            head_dim = params.get("head_dim", self.config["hidden_size"] // params.get("num_attention_heads", 4))
            return head_dim, params["rope_theta"]
        elif attention_type == "mla":
            return params.get("qk_rope_head_dim", 8), 10000.0
        elif attention_type == "csa_hca":
            mechanism = params["attention_mechanism"]
            rope_theta = params["rope_theta"] if mechanism == "sliding" else params["compress_rope_theta"]
            return params.get("rope_head_dim", 8), rope_theta
        elif attention_type == "gated_delta_net":
            return params.get("head_k_dim", 16), 10000.0
        return self.config["hidden_size"] // 4, 10000.0

    def _build_attention_mask_for_layer(self, layer_idx: int, seq_len: int, device: torch.device) -> torch.Tensor:
        layer_cfg = self.config["layers"][layer_idx]
        if layer_cfg["attention_type"] == "csa_hca":
            params = self._resolve_attention_params(layer_cfg)
            return self._build_sliding_window_attention_mask(seq_len, params["window_size"], device)
        return self._build_causal_attention_mask(seq_len, device)

    @staticmethod
    def _build_causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    @staticmethod
    def _build_sliding_window_attention_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        distance = positions[:, None] - positions[None, :]
        visible = (distance >= 0) & (distance < window_size)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask[visible] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _validate_gated_delta_net_params(params: dict) -> None:
        num_k_heads = int(params["num_k_heads"])
        num_v_heads = int(params["num_v_heads"])
        if num_k_heads <= 0 or num_v_heads <= 0:
            raise ValueError("GatedDeltaNet requires positive num_k_heads and num_v_heads")
        if num_v_heads < num_k_heads or num_v_heads % num_k_heads != 0:
            raise ValueError("GatedDeltaNet requires num_v_heads to be a multiple of num_k_heads")

    def _build_attention(self, attn_type: str, params: dict, hidden_size: int, max_seq_len: int, layer_idx: int) -> nn.Module:
        cfg = ATTENTION_REGISTRY[attn_type]
        p = {k: v["default"] for k, v in cfg["params"].items()}
        p.update(params)
        head_dim = p.get("head_dim") or (hidden_size // p["num_attention_heads"] if "num_attention_heads" in p else None)

        if attn_type == "standard":
            module = StandardAttention(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_attention_heads=p["num_attention_heads"],
                max_position_embeddings=max_seq_len,
                rope_theta=p["rope_theta"],
                num_key_value_heads=p["num_key_value_heads"],
                head_dim=head_dim,
            )
        elif attn_type == "gated":
            module = GatedAttention(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_attention_heads=p["num_attention_heads"],
                rope_theta=p["rope_theta"],
                num_key_value_heads=p["num_key_value_heads"],
                head_dim=head_dim,
                rms_norm_eps=p["rms_norm_eps"],
            )
        elif attn_type == "mla":
            module = MultiHeadLatentAttention(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_attention_heads=p["num_attention_heads"],
                q_lora_rank=p["q_lora_rank"],
                kv_lora_rank=p["kv_lora_rank"],
                qk_nope_head_dim=p["qk_nope_head_dim"],
                qk_rope_head_dim=p["qk_rope_head_dim"],
                v_head_dim=p["v_head_dim"],
            )
        elif attn_type == "gated_delta_net":
            self._validate_gated_delta_net_params(p)
            module = GatedDeltaNet(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_k_heads=p["num_k_heads"],
                num_v_heads=p["num_v_heads"],
                head_k_dim=p["head_k_dim"],
                head_v_dim=p["head_v_dim"],
                conv_kernel_size=p["conv_kernel_size"],
                layer_norm_epsilon=p["layer_norm_epsilon"],
            )
        elif attn_type == "csa_hca":
            mechanism_to_layer_type = {
                "sliding": "sliding_attention",
                "csa": "compressed_sparse_attention",
                "hca": "heavily_compressed_attention",
            }
            layer_type = mechanism_to_layer_type[p["attention_mechanism"]]
            compress_ratio = 0 if layer_type == "sliding_attention" else p["compress_ratio"]
            compress_ratios = {layer_type: compress_ratio}
            module = DeepSeekV4Attention(
                layer_idx=layer_idx,
                layer_types=(layer_type,) * (layer_idx + 1),  # 兼容设置，内部会取第 layer_idx 个 layer_type
                hidden_size=hidden_size,
                num_attention_heads=p["num_attention_heads"],
                index_num_attention_heads=p["index_num_attention_heads"],
                q_lora_rank=p["q_lora_rank"],
                o_lora_rank=p["o_lora_rank"],
                head_dim=p["head_dim"],
                index_head_dim=p["index_head_dim"],
                rope_head_dim=p["rope_head_dim"],
                compress_rope_theta=p["compress_rope_theta"],
                o_groups=p["o_groups"],
                window_size=p["window_size"],
                compress_ratios=compress_ratios,
                rms_norm_eps=1e-6,
                index_topk=p["index_topk"],
                index_score_bias_alpha=p["index_score_bias_alpha"],
                max_seq_len=max_seq_len,
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")
        initialize_architecture_lab_attention(module, attn_type)
        return TrainingAttentionAdapter(module, attn_type)

    @staticmethod
    def _coerce_bool_param(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)

    def _build_ffn(self, ffn_type: str, params: dict, hidden_size: int) -> nn.Module:
        cfg = FFN_REGISTRY[ffn_type]
        p = {k: v["default"] for k, v in cfg["params"].items()}
        p.update(params)
        
        if ffn_type == "swiglu":
            return SwiGLUFFN(hidden_size, p["intermediate_size"])
        elif ffn_type == "geglu":
            return GeGLUFFN(hidden_size, p["intermediate_size"])
        elif ffn_type == "moe":
            return MoEFFN(
                hidden_size=hidden_size,
                moe_intermediate_size=p["moe_intermediate_size"],
                n_routed_experts=p["n_routed_experts"],
                n_shared_experts=p["n_shared_experts"],
                n_activated_experts=p["n_activated_experts"],
                n_expert_groups=p["n_expert_groups"],
                n_limited_groups=p["n_limited_groups"],
                route_scale=p["route_scale"],
                use_noaux_load_balance=self._coerce_bool_param(p["use_noaux_load_balance"]),
                use_seq_aux=self._coerce_bool_param(p["use_seq_aux"]),
                seq_aux_alpha=p["seq_aux_alpha"],
                bias_update_speed=p["bias_update_speed"],
            )
        raise ValueError(f"Unknown FFN type: {ffn_type}")

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        position_ids = None
        if self._needs_position_ids:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        total_aux_loss = None
        for i, layer in enumerate(self.layers):
            position_embeddings = self.rotary_embs[i](hidden_states, position_ids)
            attention_mask = self._build_attention_mask_for_layer(i, seq_len, hidden_states.device)
            attention_mask = attention_mask.expand(batch_size, -1, -1, -1)  # (batch_size, 1, seq_len, seq_len)
            hidden_states, aux_loss = layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids=position_ids,
                padding_mask=None,
            )
            if aux_loss is not None:
                total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        main_loss = None
        if labels is not None:
            main_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = main_loss if total_aux_loss is None else main_loss + total_aux_loss
        return {"loss": loss, "main_loss": main_loss, "aux_loss": total_aux_loss, "logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _count_trainable_parameters(self, module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int | bool]:
        embedding_param_count = self._count_trainable_parameters(self.embed_tokens)
        lm_head_param_count = 0 if self.config.get("share_embedding_head", False) else self._count_trainable_parameters(self.lm_head)
        total_param_count = self.count_parameters()
        remaining_param_count = total_param_count - embedding_param_count - lm_head_param_count
        return {
            "param_count": total_param_count,
            "embedding_param_count": embedding_param_count,
            "lm_head_param_count": lm_head_param_count,
            "remaining_param_count": remaining_param_count,
            "share_embedding_head": self.config.get("share_embedding_head", False),
        }


def get_modules_info() -> dict:
    return {
        "attention_types": {
            k: {"name": v["class"].__name__, "params": v["params"]}
            for k, v in ATTENTION_REGISTRY.items()
        },
        "ffn_types": {
            k: {"name": v["class"].__name__, "params": v["params"]}
            for k, v in FFN_REGISTRY.items()
        },
    }
