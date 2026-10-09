#!/usr/bin/env python3
"""Evaluate GTA predictions with official DetectRL-ZH scenario slices."""

import argparse
import json
from pathlib import Path

from gta_repro.detectrl_zh import load_detectrl_json, parse_detection_completion
from gta_repro.evaluation import evaluate_detectrl_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="labeled DetectRL-ZH JSON array")
    parser.add_argument(
        "--result", required=True, help="JSON object with preds or completions in data order"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--invalid-fallback", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    records = load_detectrl_json(args.data)
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    labels = [record.label for record in records]
    ids = [record.sample_id for record in records]
    use_ids = ids if all(sample_id is not None for sample_id in ids) else None

    parse_sources = None
    strict_formats = None
    if "preds" in result:
        preds = [int(value) for value in result["preds"]]
    elif "completions" in result:
        parsed = [
            parse_detection_completion(text, invalid_fallback=args.invalid_fallback)
            for text in result["completions"]
        ]
        preds = [item.prediction for item in parsed]
        parse_sources = [item.source for item in parsed]
        strict_formats = [item.strict_format for item in parsed]
    else:
        raise ValueError("result must contain 'preds' or 'completions'")
    if len(preds) != len(records):
        raise ValueError(f"prediction count {len(preds)} != data count {len(records)}")
    if "labels" in result and [int(value) for value in result["labels"]] != labels:
        raise ValueError("result labels do not match the data file; refusing misaligned evaluation")

    metrics = evaluate_detectrl_predictions(
        labels,
        preds,
        ids=use_ids,
        parse_sources=parse_sources,
        strict_formats=strict_formats,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

