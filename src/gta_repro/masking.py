"""Token masks that isolate the Guess and RL segments in GTA outputs."""

from typing import Iterable, List, Sequence, Tuple

import torch


TAGS = {
    "guess": ("<guess>", "</guess>"),
    "think": ("<think>", "</think>"),
    "answer": ("<answer>", "</answer>"),
}


def find_subsequence(
    sequence: Sequence[int], subsequence: Sequence[int], start: int = 0
) -> int:
    """Return the first matching index or ``-1``."""

    if not subsequence:
        raise ValueError("subsequence must not be empty")
    last = len(sequence) - len(subsequence) + 1
    for index in range(max(start, 0), max(last, 0)):
        if list(sequence[index : index + len(subsequence)]) == list(subsequence):
            return index
    return -1


def _tag_ids(tokenizer, tag: str) -> List[int]:
    ids = tokenizer.encode(tag, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer produced no IDs for tag {tag!r}")
    return list(ids)


def segment_span(
    input_ids: Sequence[int], tokenizer, segment: str, start: int = 0
) -> Tuple[int, int]:
    """Return the half-open token span containing a segment's content."""

    if segment not in TAGS:
        raise ValueError(f"unknown segment {segment!r}; expected one of {sorted(TAGS)}")
    open_tag, close_tag = TAGS[segment]
    open_ids = _tag_ids(tokenizer, open_tag)
    close_ids = _tag_ids(tokenizer, close_tag)
    open_at = find_subsequence(input_ids, open_ids, start=start)
    if open_at < 0:
        raise ValueError(f"opening tag {open_tag!r} not found")
    content_start = open_at + len(open_ids)
    close_at = find_subsequence(input_ids, close_ids, start=content_start)
    if close_at < 0:
        raise ValueError(f"closing tag {close_tag!r} not found")
    return content_start, close_at


def build_segment_mask(
    input_ids: torch.Tensor,
    tokenizer,
    segments: Iterable[str],
    *,
    search_start: int = 0,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Build a boolean mask over content tokens for one or more GTA segments.

    The tag tokens themselves are excluded, matching Figure 2 of the paper.
    ``input_ids`` may be ``[tokens]`` or ``[batch, tokens]``.
    """

    squeeze = input_ids.ndim == 1
    if squeeze:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [tokens] or [batch, tokens]")
    if attention_mask is not None and attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    wanted = tuple(segments)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_index, row in enumerate(input_ids.detach().cpu().tolist()):
        for segment in wanted:
            begin, end = segment_span(row, tokenizer, segment, start=search_start)
            if end <= begin:
                raise ValueError(f"empty {segment!r} segment in row {row_index}")
            mask[row_index, begin:end] = True
    if attention_mask is not None:
        mask &= attention_mask.to(dtype=torch.bool, device=mask.device)
    return mask.squeeze(0) if squeeze else mask


def build_rl_mask(
    input_ids: torch.Tensor,
    tokenizer,
    completion_starts: Sequence[int],
    *,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Mask the RL objective exactly as illustrated in Figure 2.

    All completion tokens, including GTA tag tokens, are active except the
    *content* inside ``<guess>...</guess>``. If a generated sequence is
    malformed and the Guess span cannot be found, its full completion remains
    active so that the zero format reward can still supply an RL signal.
    """

    squeeze = input_ids.ndim == 1
    if squeeze:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [tokens] or [batch, tokens]")
    if len(completion_starts) != input_ids.shape[0]:
        raise ValueError("one completion start is required per input row")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    elif attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    active = attention_mask.to(dtype=torch.bool, device=input_ids.device).clone()
    for row_index, start in enumerate(completion_starts):
        active[row_index, : int(start)] = False
        row = input_ids[row_index].detach().cpu().tolist()
        try:
            begin, end = segment_span(row, tokenizer, "guess", start=int(start))
        except ValueError:
            continue
        active[row_index, begin:end] = False
    return active.squeeze(0) if squeeze else active
