import math

import torch
import torch.nn.functional as F
from torch import nn

from .swiglu import SwiGLUFFN


class DeepSeekV3MoEGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_expert_groups: int,
        n_limited_groups: int,
        route_scale: float,
        use_noaux_load_balance: bool,
    ):
        super().__init__()
        if n_routed_experts % n_expert_groups != 0:
            raise ValueError("n_routed_experts must be divisible by n_expert_groups")
        if n_limited_groups < 1 or n_limited_groups > n_expert_groups:
            raise ValueError("n_limited_groups must be in [1, n_expert_groups]")
        if n_activated_experts < 1 or n_activated_experts > n_routed_experts:
            raise ValueError("n_activated_experts must be in [1, n_routed_experts]")

        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.route_scale = route_scale
        self.n_routed_experts = n_routed_experts
        self.use_noaux_load_balance = use_noaux_load_balance
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        self.bias = nn.Parameter(torch.empty(n_routed_experts), requires_grad=False)
        self.original_scores: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores_logits = F.linear(hidden_states, self.weight, None)
        scores = scores_logits.sigmoid()
        self.original_scores = scores
        scores_for_topk = scores.clone()

        if self.use_noaux_load_balance:
            scores_for_topk = scores_for_topk + self.bias

        if self.n_groups > 1:
            scores_view = scores_for_topk.view(hidden_states.size(0), self.n_groups, -1)
            if self.use_noaux_load_balance:
                group_scores = scores_view.topk(min(2, scores_view.shape[-1]), dim=-1)[0].sum(dim=-1)
            else:
                group_scores = scores_view.amax(dim=-1)
            indices_groups = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.ones(hidden_states.size(0), self.n_groups, dtype=torch.bool, device=hidden_states.device)
            mask.scatter_(dim=1, index=indices_groups, value=False)
            scores_for_topk = scores_view.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)

        _, indices = torch.topk(scores_for_topk, self.topk, dim=-1)
        weights = scores.gather(dim=1, index=indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        weights = weights * self.route_scale
        return weights.type_as(hidden_states), indices


class MoEFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        n_activated_experts: int = 2,
        n_expert_groups: int = 4,
        n_limited_groups: int = 2,
        route_scale: float = 1.0,
        use_noaux_load_balance: bool = True,
        use_seq_aux: bool = True,
        seq_aux_alpha: float = 1e-4,
        bias_update_speed: float = 1e-3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.use_seq_aux = use_seq_aux
        self.seq_aux_alpha = seq_aux_alpha
        self.bias_update_speed = bias_update_speed
        self.gate = DeepSeekV3MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            route_scale=route_scale,
            use_noaux_load_balance=use_noaux_load_balance,
        )
        self.experts = nn.ModuleList(
            [SwiGLUFFN(hidden_size, moe_intermediate_size) for _ in range(n_routed_experts)]
        )
        self.shared_experts = SwiGLUFFN(hidden_size, n_shared_experts * moe_intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        shape = hidden_states.size()
        batch_size, seq_length = shape[:2]
        flat_states = hidden_states.view(-1, self.hidden_size)
        weights, indices = self.gate(flat_states)
        routed_output = torch.zeros_like(flat_states)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)

        if self.gate.use_noaux_load_balance and self.training:
            avg_count = counts.sum().float() / self.n_routed_experts
            error = avg_count - counts.float()
            self.gate.bias.data.add_(self.bias_update_speed * torch.sign(error))

        seq_aux_loss = None
        if self.use_seq_aux and self.training and self.gate.original_scores is not None:
            scores_for_seq_aux = self.gate.original_scores.view(batch_size, seq_length, -1)
            scores_for_seq_aux = scores_for_seq_aux / scores_for_seq_aux.sum(dim=-1, keepdim=True)
            p_i = scores_for_seq_aux.mean(dim=1)

            f_i = F.one_hot(indices.view(batch_size, -1), num_classes=self.n_routed_experts)
            f_i = f_i.sum(dim=1)
            f_i = (f_i * self.n_routed_experts) / (self.n_activated_experts * seq_length)
            seq_aux_loss = (f_i * p_i).sum() * self.seq_aux_alpha

        for expert_idx in range(self.n_routed_experts):
            if counts[expert_idx] == 0:
                continue
            token_idx, top_idx = torch.where(indices == expert_idx)
            routed_output[token_idx] += self.experts[expert_idx](flat_states[token_idx]) * weights[token_idx, top_idx, None]

        shared_output = self.shared_experts(flat_states)
        return (routed_output + shared_output).view(shape), seq_aux_loss
