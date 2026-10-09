"""Gradient-conflict rule from Equation (11) of the GTA paper."""

from typing import Dict, Iterable, List

import torch


def _trainable(parameters: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def _dot(left, right, parameters) -> torch.Tensor:
    terms = []
    for grad_left, grad_right, parameter in zip(left, right, parameters):
        if grad_left is not None and grad_right is not None:
            terms.append((grad_left.float() * grad_right.float()).sum())
    if not terms:
        device = parameters[0].device if parameters else torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.stack(terms).sum()


def backward_with_paper_conflict_rule(
    sft_loss: torch.Tensor,
    rl_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    lambda_sft: float = 1.0,
    lambda_rl: float = 1.0,
) -> Dict[str, float]:
    """Assign gradients using the paper's discard-SFT-on-conflict rule.

    If ``dot(grad_sft, grad_rl) > 0``, the weighted sum is used. Otherwise,
    only the weighted RL gradient is retained. This is the rule printed in
    Equation (11), despite the paper mentioning PCGrad more generally.
    """

    params = _trainable(parameters)
    if not params:
        raise ValueError("no trainable parameters")
    sft_grads = torch.autograd.grad(
        sft_loss, params, retain_graph=True, allow_unused=True
    )
    rl_grads = torch.autograd.grad(rl_loss, params, allow_unused=True)
    dot = _dot(sft_grads, rl_grads, params)
    use_total = bool(dot.detach().item() > 0.0)

    for parameter, grad_sft, grad_rl in zip(params, sft_grads, rl_grads):
        if grad_rl is None and (grad_sft is None or not use_total):
            parameter.grad = None
            continue
        grad = torch.zeros_like(parameter)
        if grad_rl is not None:
            grad.add_(grad_rl, alpha=lambda_rl)
        if use_total and grad_sft is not None:
            grad.add_(grad_sft, alpha=lambda_sft)
        parameter.grad = grad

    return {
        "gradient_dot": float(dot.detach().cpu()),
        "used_total_loss": float(use_total),
    }

