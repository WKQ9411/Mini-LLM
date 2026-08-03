from functools import lru_cache
import torch
from torch import nn

from mini_models.rope import compute_yarn_parameters


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)  # (total_tokens, num_heads, head_size // 2)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_type: str = "default",
        factor: float | None = None,
        attention_factor: float | None = None,
        beta_fast: float = 32,
        beta_slow: float = 1,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        if rope_type == "default":
            inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
            attention_scaling = 1.0
        elif rope_type == "yarn":
            if factor is None:
                raise ValueError("YaRN requires a scaling factor")
            inv_freq, attention_scaling = compute_yarn_parameters(
                base,
                {
                    "rope_type": "yarn",
                    "factor": factor,
                    "attention_factor": attention_factor,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                },
                rotary_dim,
                max_position_embeddings,  # 扩展后的最大位置编码长度
            )
        else:
            raise ValueError(f"Unsupported rope_type: {rope_type}")

        t = torch.arange(max_position_embeddings, dtype=torch.float)  # 绝对位置编码的索引
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # (max_position_embeddings, rotary_dim // 2) 计算每个位置的旋转角度
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)  # (max_position_embeddings, 1, rotary_dim)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]  # 取出对应位置的 cos 和 sin，(total_tokens, 1, rotary_dim)
        cos, sin = cos_sin.chunk(2, dim=-1)  # (total_tokens, 1, rotary_dim // 2)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_type: str = "default",
    factor: float | None = None,
    attention_factor: float | None = None,
    beta_fast: float = 32,
    beta_slow: float = 1,
):
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        rope_type,
        factor,
        attention_factor,
        beta_fast,
        beta_slow,
    )
    return rotary_emb
