"""Masked SFT and GRPO objectives for GTA."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def causal_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return next-token log-probabilities aligned to ``input_ids[:, 1:]``."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [batch,tokens,vocab] and input_ids [batch,tokens]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must agree in batch and sequence dimensions")
    shifted = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return shifted.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def masked_causal_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Causal cross-entropy averaged only over selected target tokens."""

    logps = causal_token_logps(logits, input_ids)
    active = token_mask[:, 1:].to(dtype=logps.dtype, device=logps.device)
    denom = active.sum().clamp_min(1.0)
    return -(logps * active).sum() / denom


def normalized_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, group]")
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / std.clamp_min(eps)


def grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    beta: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the masked GRPO loss from Equations (5)-(8).

    Log-probability tensors have shape ``[batch, group, tokens]`` and the
    token mask has the same shape. The implemented KL estimator is exactly
    ``p_ref/p_theta - log(p_ref/p_theta) - 1`` from Equations (6)-(7).
    """

    if not (
        policy_logps.shape == old_logps.shape == ref_logps.shape == token_mask.shape
    ):
        raise ValueError("all log-probabilities and token_mask must have equal shapes")
    if policy_logps.ndim != 3 or rewards.shape != policy_logps.shape[:2]:
        raise ValueError("expected logps [batch,group,tokens] and rewards [batch,group]")

    advantages = normalized_group_advantages(rewards).unsqueeze(-1)
    ratio = torch.exp(policy_logps - old_logps)
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)

    log_ratio_ref = ref_logps - policy_logps
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    objective = surrogate - beta * kl

    active = token_mask.to(dtype=objective.dtype, device=objective.device)
    per_sequence = (objective * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1.0)
    loss = -per_sequence.mean()
    metrics = {
        "advantage_mean": advantages.mean().detach(),
        "kl_mean": ((kl * active).sum() / active.sum().clamp_min(1.0)).detach(),
        "ratio_mean": ((ratio * active).sum() / active.sum().clamp_min(1.0)).detach(),
    }
    return loss, metrics

