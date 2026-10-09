"""Official-style DetectRL-ZH metrics and scenario slices."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


MAIN_SCENARIOS: Dict[str, Tuple[int, int]] = {
    "Normal": (1, 4000),
    "Attack": (4001, 7000),
    "Varying Lengths": (7001, 11000),
}

ATTACK_SCENARIOS: Dict[str, Tuple[int, int]] = {
    "Attack-Mixed": (4001, 5000),
    "Attack-Paraphrase": (5001, 6000),
    "Attack-Perturbation": (6001, 7000),
}

VARYING_SCENARIOS: Dict[str, Tuple[int, int]] = {
    "Varying-64": (7001, 8000),
    "Varying-128": (8001, 9000),
    "Varying-256": (9001, 10000),
    "Varying-512": (10001, 11000),
}


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, object]:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred) or not len(true):
        raise ValueError("y_true and y_pred must be non-empty aligned 1-D arrays")
    if not set(np.unique(true)).issubset({0, 1}) or not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("DetectRL-ZH metrics require binary labels 0/1")
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "n_examples": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(np.mean(f1)),
        "human_f1": float(f1[0]),
        "ai_f1": float(f1[1]),
        "human_precision": float(precision[0]),
        "ai_precision": float(precision[1]),
        "human_recall": float(recall[0]),
        "ai_recall": float(recall[1]),
        "human_support": int(support[0]),
        "ai_support": int(support[1]),
        "confusion_matrix": confusion_matrix(true, pred, labels=[0, 1]).tolist(),
    }


def _slice_metrics(true, pred, ids, ranges):
    output = {}
    for name, (lower, upper) in ranges.items():
        mask = (ids >= lower) & (ids <= upper)
        if mask.any():
            output[name] = binary_metrics(true[mask], pred[mask])
    return output


def evaluate_detectrl_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    ids: Optional[Sequence[int]] = None,
    parse_sources: Optional[Sequence[str]] = None,
    strict_formats: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Evaluate overall and official ID-based DetectRL-ZH scenarios."""

    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    result: Dict[str, object] = {"test_overall": binary_metrics(true, pred)}

    if parse_sources is not None:
        if len(parse_sources) != len(true):
            raise ValueError("parse_sources must align with predictions")
        counts = {name: list(parse_sources).count(name) for name in ("answer", "guess", "fallback")}
        result["generation_parse"] = {
            "counts": counts,
            "answer_rate": counts["answer"] / len(true),
            "fallback_rate": counts["fallback"] / len(true),
        }
    if strict_formats is not None:
        if len(strict_formats) != len(true):
            raise ValueError("strict_formats must align with predictions")
        result.setdefault("generation_parse", {})["strict_format_rate"] = float(
            np.mean(np.asarray(strict_formats, dtype=bool))
        )

    if ids is not None:
        sample_ids = np.asarray(ids, dtype=int)
        if sample_ids.shape != true.shape:
            raise ValueError("ids must align with predictions")
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be unique")
        main = _slice_metrics(true, pred, sample_ids, MAIN_SCENARIOS)
        attack = _slice_metrics(true, pred, sample_ids, ATTACK_SCENARIOS)
        varying = _slice_metrics(true, pred, sample_ids, VARYING_SCENARIOS)
        result["subtask_results"] = {
            "main_subtasks": main,
            "attack_detail": attack,
            "varying_detail": varying,
            "avg_f1": float(np.mean([item["macro_f1"] for item in main.values()]))
            if len(main) == len(MAIN_SCENARIOS)
            else None,
        }
    return result

