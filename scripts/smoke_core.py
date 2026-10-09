#!/usr/bin/env python3
"""Run a dependency-light numerical smoke test of the GTA core."""

import json

import torch

from gta_repro.gradients import backward_with_paper_conflict_rule
from gta_repro.losses import grpo_loss
from gta_repro.rewards import compute_rewards


def main() -> None:
    completions = [[
        "<guess>positive</guess><think>good tone</think><answer>positive</answer>",
        "<guess>negative</guess><think>bad tone</think><answer>negative</answer>",
    ]]
    rewards = compute_rewards(completions, ["positive"], ["negative", "positive"])

    policy = torch.tensor([[[-0.8, -0.7], [-0.6, -0.9]]], requires_grad=True)
    old = policy.detach().clone()
    ref = policy.detach().clone() - 0.05
    mask = torch.ones_like(policy, dtype=torch.bool)
    rl_loss, rl_metrics = grpo_loss(policy, old, ref, rewards, mask)
    rl_loss.backward()

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    sft_conflicting = (parameter - 1.0).pow(2)
    rl_conflicting = (parameter + 1.0).pow(2)
    conflict = backward_with_paper_conflict_rule(
        sft_conflicting, rl_conflicting, [parameter]
    )
    conflict_gradient = float(parameter.grad)

    parameter.grad = None
    sft_aligned = (parameter + 2.0).pow(2)
    rl_aligned = (parameter + 1.0).pow(2)
    aligned = backward_with_paper_conflict_rule(sft_aligned, rl_aligned, [parameter])
    aligned_gradient = float(parameter.grad)

    result = {
        "rewards": rewards.tolist(),
        "grpo_loss": float(rl_loss.detach()),
        "grpo_policy_grad": policy.grad.tolist(),
        "grpo_metrics": {key: float(value) for key, value in rl_metrics.items()},
        "conflict": {**conflict, "assigned_gradient": conflict_gradient},
        "aligned": {**aligned, "assigned_gradient": aligned_gradient},
    }
    print(json.dumps(result, indent=2))

    assert rewards.tolist() == [[2.0, 1.0]]
    assert conflict["used_total_loss"] == 0.0 and conflict_gradient > 0
    assert aligned["used_total_loss"] == 1.0 and aligned_gradient > 0
    assert torch.isfinite(rl_loss)


if __name__ == "__main__":
    main()

