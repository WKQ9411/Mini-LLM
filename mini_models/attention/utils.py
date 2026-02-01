import torch

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 key 和 value 以匹配 query 头数, 与 torch.repeat_interleave 等价, 但不用复制数据, 更高效

    Args:
        hidden_states (torch.Tensor): 输入张量 (batch_size, n_kv_heads, seq_len, head_dim)
        n_rep (int): 重复次数

    Returns:
        torch.Tensor: 输出张量 (batch_size, n_heads, seq_len, head_dim)
    """
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.size()

    if n_rep == 1:
        return hidden_states

    return (
        hidden_states[:, :, None, :, :]
        .expand(bsz, num_key_value_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)
    )