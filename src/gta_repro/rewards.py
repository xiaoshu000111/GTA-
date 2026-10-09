"""Rule-based rewards from Equations (2)-(4) of the GTA paper."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

from .formatting import parse_gta_output


@dataclass(frozen=True)
class RewardBreakdown:
    format_reward: float
    accuracy_reward: float

    @property
    def total(self) -> float:
        return self.format_reward + self.accuracy_reward


def score_completion(
    completion: str,
    gold_label: str,
    valid_labels: Iterable[str],
) -> RewardBreakdown:
    parsed = parse_gta_output(completion)
    label_set = {str(label).strip() for label in valid_labels}
    format_ok = (
        parsed is not None
        and bool(parsed.think)
        and parsed.guess in label_set
        and parsed.answer in label_set
    )
    accurate = parsed is not None and parsed.answer == str(gold_label).strip()
    return RewardBreakdown(float(format_ok), float(accurate))


def compute_rewards(
    completions: Sequence[Sequence[str]],
    gold_labels: Sequence[str],
    valid_labels: Iterable[str],
    *,
    device: torch.device = None,
) -> torch.Tensor:
    """Return rewards with shape ``[batch, group]``."""

    if len(completions) != len(gold_labels):
        raise ValueError("completions and gold_labels must have the same batch size")
    rows: List[List[float]] = []
    for group, gold in zip(completions, gold_labels):
        rows.append(
            [score_completion(item, str(gold), valid_labels).total for item in group]
        )
    if not rows or not rows[0]:
        raise ValueError("at least one prompt and one completion per prompt are required")
    group_size = len(rows[0])
    if any(len(row) != group_size for row in rows):
        raise ValueError("all prompts must have the same number of completions")
    return torch.tensor(rows, dtype=torch.float32, device=device)

