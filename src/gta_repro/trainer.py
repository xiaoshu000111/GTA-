"""A transparent single-update GTA reference implementation.

This module deliberately exposes a single optimizer update rather than hiding
the algorithm in a large trainer. It is suitable for unit tests, small-scale
experiments, and integration into DeepSpeed/Accelerate training loops.
"""

from typing import Dict, Iterable, Optional, Sequence

import torch

from .gradients import backward_with_paper_conflict_rule
from .losses import causal_token_logps, grpo_loss, masked_causal_ce
from .masking import build_rl_mask, build_segment_mask
from .rewards import compute_rewards


def _batch_tokenize(tokenizer, texts, max_length, device):
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def gta_optimizer_step(
    *,
    model,
    tokenizer,
    optimizer,
    prompts: Sequence[str],
    completion_groups: Sequence[Sequence[str]],
    gold_labels: Sequence[str],
    valid_labels: Iterable[str],
    reference_model=None,
    max_length: int = 4096,
    clip_epsilon: float = 0.2,
    beta: float = 0.01,
    lambda_sft: float = 1.0,
    lambda_rl: float = 1.0,
    max_grad_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """Perform one GTA update on already-generated completion groups.

    The current policy before the update is the GRPO old policy. Passing a
    frozen ``reference_model`` enables the paper's KL penalty. To reproduce
    reuse factor > 1 exactly, a surrounding rollout buffer should retain the
    old log-probabilities across repeated calls; this one-step reference keeps
    reuse at 1 by design.
    """

    if not prompts or len(prompts) != len(completion_groups):
        raise ValueError("prompts and completion_groups must form a non-empty batch")
    if len(prompts) != len(gold_labels):
        raise ValueError("one gold label is required per prompt")
    group_size = len(completion_groups[0])
    if group_size < 2 or any(len(group) != group_size for group in completion_groups):
        raise ValueError("each prompt needs the same group size of at least two")

    # Keep a whitespace boundary before the first GTA tag. Without it, BPE
    # tokenizers can merge the prompt's final punctuation with "<guess>",
    # making token-level tag discovery ambiguous.
    prompts = [prompt if prompt[-1:].isspace() else prompt + "\n" for prompt in prompts]
    device = next(model.parameters()).device
    flat_prompts = [prompt for prompt in prompts for _ in range(group_size)]
    flat_completions = [item for group in completion_groups for item in group]
    full_texts = [prompt + completion for prompt, completion in zip(flat_prompts, flat_completions)]
    prompt_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in flat_prompts
    ]
    rl_batch = _batch_tokenize(tokenizer, full_texts, max_length, device)
    rl_mask = build_rl_mask(
        rl_batch["input_ids"],
        tokenizer,
        prompt_lengths,
        attention_mask=rl_batch.get("attention_mask"),
    )

    outputs = model(**rl_batch)
    policy_logps_flat = causal_token_logps(outputs.logits, rl_batch["input_ids"])
    old_logps_flat = policy_logps_flat.detach().clone()
    if reference_model is None:
        ref_logps_flat = old_logps_flat
    else:
        with torch.no_grad():
            ref_outputs = reference_model(**rl_batch)
            ref_logps_flat = causal_token_logps(ref_outputs.logits, rl_batch["input_ids"])

    token_count = policy_logps_flat.shape[-1]
    shape = (len(prompts), group_size, token_count)
    rewards = compute_rewards(
        completion_groups, gold_labels, valid_labels, device=device
    )
    rl_loss, rl_metrics = grpo_loss(
        policy_logps_flat.reshape(shape),
        old_logps_flat.reshape(shape),
        ref_logps_flat.reshape(shape),
        rewards,
        rl_mask[:, 1:].reshape(shape),
        clip_epsilon=clip_epsilon,
        beta=beta,
    )

    sft_completions = [f"<guess>{label}</guess>" for label in gold_labels]
    sft_texts = [prompt + completion for prompt, completion in zip(prompts, sft_completions)]
    sft_starts = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    sft_batch = _batch_tokenize(tokenizer, sft_texts, max_length, device)
    sft_mask = build_segment_mask(
        sft_batch["input_ids"],
        tokenizer,
        ["guess"],
        search_start=min(sft_starts),
        attention_mask=sft_batch.get("attention_mask"),
    )
    sft_outputs = model(**sft_batch)
    sft_loss = masked_causal_ce(
        sft_outputs.logits, sft_batch["input_ids"], sft_mask
    )

    optimizer.zero_grad(set_to_none=True)
    gradient_stats = backward_with_paper_conflict_rule(
        sft_loss,
        rl_loss,
        model.parameters(),
        lambda_sft=lambda_sft,
        lambda_rl=lambda_rl,
    )
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_grad_norm,
        )
    optimizer.step()

    return {
        "sft_loss": float(sft_loss.detach().cpu()),
        "rl_loss": float(rl_loss.detach().cpu()),
        "reward_mean": float(rewards.mean().detach().cpu()),
        **gradient_stats,
        **{key: float(value.cpu()) for key, value in rl_metrics.items()},
    }
