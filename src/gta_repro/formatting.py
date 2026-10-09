"""Prompt and output-format utilities for Guess-Think-Answer (GTA)."""

from dataclasses import dataclass
import re
from typing import Iterable, Optional


_GTA_PATTERN = re.compile(
    r"^\s*<guess>\s*(?P<guess>.*?)\s*</guess>\s*"
    r"<think>\s*(?P<think>.*?)\s*</think>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class GTAOutput:
    guess: str
    think: str
    answer: str


def parse_gta_output(text: str) -> Optional[GTAOutput]:
    """Parse a complete GTA output; return ``None`` for malformed output."""

    match = _GTA_PATTERN.fullmatch(text)
    if not match:
        return None
    return GTAOutput(
        guess=match.group("guess").strip(),
        think=match.group("think").strip(),
        answer=match.group("answer").strip(),
    )


def build_prompt(text: str, labels: Iterable[str]) -> str:
    """Build the classification instruction described in Figure 1 of the paper."""

    label_list = ", ".join(str(label) for label in labels)
    return (
        "Please perform a text classification. The user inputs text, and the "
        f"assistant selects a label from {{{label_list}}} as output. "
        "The assistant first makes an initial guess, then thinks about whether "
        "the guessed answer is correct based on the user's input. Finally, based "
        "on the reasoning process, the assistant provides the final answer. "
        "Use exactly this format: <guess>label</guess><think>reasoning</think>"
        "<answer>label</answer>. Only return a label inside guess and answer.\n"
        f"Text: {text}\nAssistant:\n"
    )
