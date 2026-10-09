"""DetectRL-ZH task adapter for GTA.

The task is document-level binary classification:

* 0: human-written text
* 1: machine-generated text

It is not span attribution, misconduct adjudication, or estimation of the
percentage of AI involvement in a document.
"""

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .formatting import parse_gta_output


HUMAN_LABEL = 0
MACHINE_LABEL = 1
LABEL_ID_TO_TEXT = {
    HUMAN_LABEL: "人类撰写",
    MACHINE_LABEL: "机器生成",
}
LABEL_TEXT_TO_ID = {text: label for label, text in LABEL_ID_TO_TEXT.items()}
DETECTRL_LABELS = tuple(LABEL_ID_TO_TEXT.values())


SYSTEM_PROMPT = (
    "你是中文大模型生成文本检测器。你的任务是判断整篇输入文本更符合人类撰写"
    "还是机器生成。该判断只表示文档级来源分类，不代表学术违规结论，也不表示"
    "文档中AI内容所占百分比。请先给出初步判断，再简要检查语言模式与连贯性，"
    "最后给出最终判断。"
)

USER_INSTRUCTION = (
    "请对下面的中文文本进行二分类。只能从“人类撰写”和“机器生成”中选择标签，"
    "并严格输出：<guess>标签</guess><think>简短理由</think><answer>标签</answer>。"
)


@dataclass(frozen=True)
class DetectRLRecord:
    text: str
    label: Optional[int] = None
    sample_id: Optional[int] = None
    model: Optional[str] = None
    source: Optional[str] = None

    @property
    def gold_text(self) -> str:
        if self.label is None:
            raise ValueError("unlabeled record has no gold label text")
        return LABEL_ID_TO_TEXT[self.label]


@dataclass(frozen=True)
class ParsedDetection:
    prediction: int
    source: str
    strict_format: bool
    guess: str
    answer: str


def build_detectrl_prompt(text: str, tokenizer=None) -> str:
    """Build a DetectRL-ZH GTA prompt, optionally using a chat template."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    user_content = f"{USER_INSTRUCTION}\n\n待检测文本：\n{text}"
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"系统：{SYSTEM_PROMPT}\n用户：{user_content}\n助手：\n"


def label_to_text(label: int) -> str:
    try:
        return LABEL_ID_TO_TEXT[int(label)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"DetectRL-ZH label must be 0 or 1, got {label!r}") from exc


def parse_detection_completion(
    completion: str, *, invalid_fallback: int = HUMAN_LABEL
) -> ParsedDetection:
    """Map a GTA completion to a binary prediction with auditable fallbacks.

    A valid final Answer is always preferred. A valid Guess is used only when
    the full format or final Answer is invalid. Fully unparseable output uses
    ``invalid_fallback`` and is reported with ``source='fallback'``.
    """

    fallback_text = label_to_text(invalid_fallback)
    parsed = parse_gta_output(completion)
    if parsed is not None and parsed.answer in LABEL_TEXT_TO_ID:
        return ParsedDetection(
            prediction=LABEL_TEXT_TO_ID[parsed.answer],
            source="answer",
            strict_format=(
                bool(parsed.think)
                and parsed.guess in LABEL_TEXT_TO_ID
                and parsed.answer in LABEL_TEXT_TO_ID
            ),
            guess=parsed.guess,
            answer=parsed.answer,
        )

    # A malformed closing section may still contain an unambiguous Guess.
    guess = parsed.guess if parsed is not None else ""
    if not guess:
        import re

        match = re.search(r"<guess>\s*(.*?)\s*</guess>", completion, flags=re.DOTALL)
        guess = match.group(1).strip() if match else ""
    if guess in LABEL_TEXT_TO_ID:
        return ParsedDetection(
            prediction=LABEL_TEXT_TO_ID[guess],
            source="guess",
            strict_format=False,
            guess=guess,
            answer=parsed.answer if parsed is not None else "",
        )
    return ParsedDetection(
        prediction=LABEL_TEXT_TO_ID[fallback_text],
        source="fallback",
        strict_format=False,
        guess=guess,
        answer=parsed.answer if parsed is not None else "",
    )


def _record_from_dict(item: Dict, *, require_label: bool) -> DetectRLRecord:
    if "text" not in item or not isinstance(item["text"], str) or not item["text"].strip():
        raise ValueError("each DetectRL-ZH record needs a non-empty string field 'text'")
    label = item.get("label")
    if label is None:
        if require_label:
            raise ValueError("labeled DetectRL-ZH data requires field 'label'")
    else:
        try:
            label = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid label {label!r}") from exc
        label_to_text(label)
    sample_id = item.get("id")
    if sample_id is not None:
        sample_id = int(sample_id)
    return DetectRLRecord(
        text=item["text"],
        label=label,
        sample_id=sample_id,
        model=str(item["model"]) if item.get("model") is not None else None,
        source=str(item["source"]) if item.get("source") is not None else None,
    )


def load_detectrl_json(
    path: str,
    *,
    require_label: bool = True,
    max_samples: Optional[int] = None,
) -> List[DetectRLRecord]:
    """Load the official JSON-array format without altering text content."""

    with Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("DetectRL-ZH file must contain a JSON array")
    records = [_record_from_dict(item, require_label=require_label) for item in data]
    return records[:max_samples] if max_samples is not None else records


def summarize_records(records: Sequence[DetectRLRecord]) -> Dict[str, object]:
    labels: Dict[str, int] = {"human": 0, "machine": 0, "unlabeled": 0}
    models: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for record in records:
        if record.label == HUMAN_LABEL:
            labels["human"] += 1
        elif record.label == MACHINE_LABEL:
            labels["machine"] += 1
        else:
            labels["unlabeled"] += 1
        if record.model is not None:
            models[record.model] = models.get(record.model, 0) + 1
        if record.source is not None:
            sources[record.source] = sources.get(record.source, 0) + 1
    return {
        "total": len(records),
        "labels": labels,
        "models": dict(sorted(models.items())),
        "sources": dict(sorted(sources.items())),
    }


def downsample_binary_balance(
    records: Sequence[DetectRLRecord], *, seed: int = 42
) -> List[DetectRLRecord]:
    """Downsample the majority class to create a reproducible 1:1 set."""

    human = [record for record in records if record.label == HUMAN_LABEL]
    machine = [record for record in records if record.label == MACHINE_LABEL]
    if not human or not machine:
        raise ValueError("binary balancing requires both human and machine samples")
    size = min(len(human), len(machine))
    rng = random.Random(seed)
    balanced = rng.sample(human, size) + rng.sample(machine, size)
    rng.shuffle(balanced)
    return balanced

