import warnings

import torch
from torch import nn


# Inductor 在词表 Softmax 被拆分归约时会输出性能提示，但不影响计算正确性
warnings.filterwarnings(
    "ignore",
    message=r"Online softmax is disabled on the fly",
    category=UserWarning,
    module=r"torch\._inductor\.lowering",
)


class Sampler(nn.Module):

    @torch.compile(dynamic=True)
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_ks: torch.Tensor | None,
        top_ps: torch.Tensor | None,
        repetition_penalties: torch.Tensor | None,
        frequency_penalties: torch.Tensor | None,
        penalty_token_ids: torch.Tensor | None,
        penalty_token_counts: torch.Tensor | None,
        has_greedy: bool,
        all_greedy: bool,
        use_temperature: bool,
        use_filter: bool,
        use_penalty: bool,
    ) -> torch.Tensor:
        # logits: (num_seqs, vocab_size)
        if all_greedy and not use_penalty:
            return logits.argmax(dim=-1)

        logits = logits.float()

        if use_penalty:
            assert repetition_penalties is not None
            assert frequency_penalties is not None
            assert penalty_token_ids is not None
            assert penalty_token_counts is not None

            # 重复惩罚
            # penalty_token_ids 每行只保存该序列已经生成过的唯一 token，-1 是 padding
            valid_penalty_tokens = penalty_token_ids >= 0  # 有效掩码 (num_seqs, max_unique_tokens)
            safe_penalty_token_ids = penalty_token_ids.clamp_min(0)  # 将 -1 替换为 0，避免 gather 时索引越界
            penalty_scores = logits.gather(1, safe_penalty_token_ids)  # 获取每个序列中已经生成过的 token 的 logits (num_seqs, max_unique_tokens)

            repetition_penalties = repetition_penalties.unsqueeze(1)  # (num_seqs, 1)
            repeated_scores = torch.where(
                penalty_scores < 0,
                penalty_scores * repetition_penalties,  # 如果 logits < 0，则乘以 repetition_penalty，让其更负
                penalty_scores / repetition_penalties,  # 如果 logits > 0，则除以 repetition_penalty，让其更小
            )  # (num_seqs, max_unique_tokens)
            repetition_deltas = torch.where(
                valid_penalty_tokens,
                repeated_scores - penalty_scores,  # 计算每个序列中已经生成过的 token 的 logits 变化量
                torch.zeros_like(penalty_scores),  # 对于 padding token，变化量为 0
            )  # (num_seqs, max_unique_tokens)
            logits.scatter_add_(1, safe_penalty_token_ids, repetition_deltas)  # 将变化量加回 logits 中，更新已经生成过的 token 的 logits

            # 频率惩罚
            frequency_deltas = -penalty_token_counts * frequency_penalties.unsqueeze(1)  # 计算每个序列中已经生成过的 token 的频率惩罚变化量 (num_seqs, max_unique_tokens)
            frequency_deltas = torch.where(
                valid_penalty_tokens,
                frequency_deltas,  # 对于有效 token，使用计算出的频率惩罚变化量
                torch.zeros_like(frequency_deltas),  # 对于 padding token，变化量为 0
            )
            logits.scatter_add_(1, safe_penalty_token_ids, frequency_deltas)  # 将频率惩罚变化量加回 logits 中，更新已经生成过的 token 的 logits

        if all_greedy:
            return logits.argmax(dim=-1)  # 如果所有序列都要求贪婪采样，则直接返回 greedy_tokens

        greedy_tokens = logits.argmax(dim=-1) if has_greedy else None  # 如果有序列要求贪婪采样，则计算 greedy_tokens，否则为 None
        sampling_logits = logits
        if use_temperature:
            assert temperatures is not None
            safe_temperatures = torch.where(temperatures > 0, temperatures, torch.ones_like(temperatures)) if has_greedy else temperatures
            sampling_logits = logits.div(safe_temperatures.unsqueeze(1))  # 应用温度 (num_seqs, vocab_size)

        if not use_filter:
            # 未启用 top-k/top-p 时直接在原始词表概率上采样，跳过排序、累积概率和索引映射
            probs = sampling_logits.softmax(dim=-1)
            noise = torch.empty_like(probs).exponential_().clamp_min_(1e-10)
            sampled_tokens = probs.div_(noise).argmax(dim=-1)
            if has_greedy:
                assert temperatures is not None and greedy_tokens is not None
                return torch.where(temperatures == 0, greedy_tokens, sampled_tokens)
            return sampled_tokens

        # top-k
        assert top_ks is not None and top_ps is not None
        sorted_logits, sorted_indices = sampling_logits.sort(dim=-1, descending=True)  # 按照降序排列 (num_seqs, vocab_size), (num_seqs, vocab_size)
        vocab_size = sorted_logits.size(1)
        effective_top_ks = torch.where(
            top_ks > 0,
            top_ks.clamp_max(vocab_size),  # 如果 top_k > vocab_size，则取 vocab_size
            torch.full_like(top_ks, vocab_size),  # 如果 top_k <= 0，则取 vocab_size，表示不限制 top_k
        )  # (num_seqs,)
        top_k_thresholds = sorted_logits.gather(1, (effective_top_ks - 1).unsqueeze(1))  # 获取每个序列中第 k 大 logit，作为 top-k 截断阈值 (num_seqs, 1)
        sorted_logits.masked_fill_(sorted_logits < top_k_thresholds, -float("inf"))  # 将小于 top-k 阈值的 logits 设置为 -inf，表示这些 token 不会被采样

        # top-p
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # 计算累积概率 (num_seqs, vocab_size)
        top_p_mask = cumulative_probs > top_ps.unsqueeze(1)  # 获取每个序列中累积概率大于 top-p 的 token 的掩码 (num_seqs, vocab_size)
        shifted_top_p_mask = torch.zeros_like(top_p_mask)  # 创建一个与 top_p_mask 相同形状的全零张量，用于存储 top-p 掩码的右移版本
        shifted_top_p_mask[:, 1:] = top_p_mask[:, :-1]  # 将 top_p_mask 右移一位，表示保留第一个大于 top-p 的 token，同时将其余 token 标记为 True (num_seqs, vocab_size)
        top_p_enabled = (top_ps > 0.0) & (top_ps < 1.0)  # 判断每个序列是否启用 top-p 采样 (num_seqs,)
        sorted_logits.masked_fill_(shifted_top_p_mask & top_p_enabled.unsqueeze(1), -float("inf"))  # 将大于 top-p 且启用 top-p 的 token 的 logits 设置为 -inf，表示这些 token 不会被采样

        probs = sorted_logits.softmax(dim=-1)  # 计算最终的采样概率 (num_seqs, vocab_size)
        # 首先创建一个和 probs 形状相同的张量，然后原地填入指数分布的噪声，并把噪声夹到最小 1e-10，防止除零
        # div_ 是原地除法，这是 Gumbel-max 采样技巧：用概率除以一个随机噪声再取 argmax 来实现按概率采样一个 token
        noise = torch.empty_like(probs).exponential_().clamp_min_(1e-10)  # 生成指数分布噪声，并将其最小值限制为 1e-10，避免除零 (num_seqs, vocab_size)
        sampled_ranks = probs.div_(noise).argmax(dim=-1, keepdim=True)  # 按概率采样一个 token 的索引 (num_seqs, 1)
        sampled_tokens = sorted_indices.gather(1, sampled_ranks).squeeze(1)  # 将采样的索引映射回原始的 token id (num_seqs,)
        if has_greedy:
            assert temperatures is not None and greedy_tokens is not None
            return torch.where(temperatures == 0, greedy_tokens, sampled_tokens)  # 如果温度为 0，则返回贪婪采样的 token，否则返回按概率采样的 token
        return sampled_tokens
