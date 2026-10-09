"""Core components for the GTA text-classification reproduction."""

from .formatting import GTAOutput, build_prompt, parse_gta_output
from .detectrl_zh import (
    DETECTRL_LABELS,
    build_detectrl_prompt,
    load_detectrl_json,
    parse_detection_completion,
)
from .evaluation import evaluate_detectrl_predictions
from .gradients import backward_with_paper_conflict_rule
from .losses import causal_token_logps, grpo_loss, masked_causal_ce
from .masking import build_rl_mask, build_segment_mask, find_subsequence
from .rewards import compute_rewards

__all__ = [
    "GTAOutput",
    "DETECTRL_LABELS",
    "backward_with_paper_conflict_rule",
    "build_prompt",
    "build_detectrl_prompt",
    "build_rl_mask",
    "build_segment_mask",
    "causal_token_logps",
    "compute_rewards",
    "find_subsequence",
    "grpo_loss",
    "evaluate_detectrl_predictions",
    "load_detectrl_json",
    "masked_causal_ce",
    "parse_gta_output",
    "parse_detection_completion",
]
