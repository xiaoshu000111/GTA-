#!/usr/bin/env python3
"""Run resumable generative inference and official DetectRL-ZH evaluation.

The script writes one JSON object per sample before moving to the next batch,
so an interrupted Colab session can resume without regenerating completed
predictions.  Gold labels are used only after generation for metrics.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from gta_repro.detectrl_zh import (
    build_detectrl_prompt,
    load_detectrl_json,
    parse_detection_completion,
)
from gta_repro.evaluation import evaluate_detectrl_predictions


def _read_progress(path, records):
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) > len(records):
        raise ValueError("prediction progress is longer than the evaluation data")
    for index, row in enumerate(rows):
        record = records[index]
        if int(row["index"]) != index:
            raise ValueError("prediction progress must contain contiguous ordered indices")
        if row.get("sample_id") != record.sample_id or int(row["label"]) != record.label:
            raise ValueError("prediction progress does not align with the evaluation data")
    return rows


def _write_confusion_csv(path, matrix):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", "human_0", "machine_1"])
        writer.writerow(["human_0", *matrix[0]])
        writer.writerow(["machine_1", *matrix[1]])


def _write_report(path, metrics, config, elapsed_seconds):
    overall = metrics["test_overall"]
    matrix = overall["confusion_matrix"]
    lines = [
        "# DetectRL-ZH evaluation",
        "",
        f"- Model: {config['model']}",
        f"- Adapter: {config['adapter'] or 'none'}",
        f"- Samples: {overall['n_examples']}",
        f"- Accuracy: {overall['accuracy']:.6f}",
        f"- Macro-F1: {overall['macro_f1']:.6f}",
        f"- Human F1: {overall['human_f1']:.6f}",
        f"- AI F1: {overall['ai_f1']:.6f}",
        f"- Strict format rate: {metrics['generation_parse']['strict_format_rate']:.6f}",
        f"- Fallback rate: {metrics['generation_parse']['fallback_rate']:.6f}",
        f"- Elapsed seconds: {elapsed_seconds:.1f}",
        "",
        "## Confusion matrix",
        "",
        "Rows are actual labels and columns are predicted labels.",
        "",
        "| actual \\ predicted | human (0) | machine (1) |",
        "|---|---:|---:|",
        f"| human (0) | {matrix[0][0]} | {matrix[0][1]} |",
        f"| machine (1) | {matrix[1][0]} | {matrix[1][1]} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Resumable Qwen GTA inference on labeled DetectRL-ZH data"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", help="LoRA adapter directory; omit for base-model baseline")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--invalid-fallback", type=int, choices=[0, 1], default=0)
    parser.add_argument("--limit", type=int, help="prefix size for a declared smoke/throughput run")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("--batch-size and --max-new-tokens must be positive")
    if args.max_length <= args.max_new_tokens:
        raise ValueError("--max-length must exceed --max-new-tokens")

    records = load_detectrl_json(args.data, max_samples=args.limit)
    if not records or any(record.label is None for record in records):
        raise ValueError("evaluation requires non-empty labeled data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "predictions.jsonl"
    progress = _read_progress(progress_path, records)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.to(device)
    model.eval()

    config = vars(args).copy()
    config.update(
        {
            "dtype": str(dtype),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "n_examples": len(records),
            "deterministic_decode": True,
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    started = time.time()
    for start in range(len(progress), len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        prompts = [
            build_detectrl_prompt(record.text[: args.max_chars], tokenizer)
            for record in batch_records
        ]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_length - args.max_new_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )
        batch_rows = []
        for offset, (record, completion) in enumerate(zip(batch_records, completions)):
            parsed = parse_detection_completion(
                completion.strip(), invalid_fallback=args.invalid_fallback
            )
            batch_rows.append(
                {
                    "index": start + offset,
                    "sample_id": record.sample_id,
                    "label": record.label,
                    "prediction": parsed.prediction,
                    "parse_source": parsed.source,
                    "strict_format": parsed.strict_format,
                    "guess": parsed.guess,
                    "answer": parsed.answer,
                    "completion": completion.strip(),
                }
            )
        with progress_path.open("a", encoding="utf-8") as stream:
            for row in batch_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        progress.extend(batch_rows)
        state = {
            "completed": len(progress),
            "total": len(records),
            "elapsed_seconds": time.time() - started,
            "max_gpu_memory_gb": (
                round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
                if device.type == "cuda"
                else None
            ),
        }
        (output_dir / "progress.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(state, ensure_ascii=False), flush=True)

    labels = [int(row["label"]) for row in progress]
    predictions = [int(row["prediction"]) for row in progress]
    ids = [row["sample_id"] for row in progress]
    parse_sources = [row["parse_source"] for row in progress]
    strict_formats = [bool(row["strict_format"]) for row in progress]
    metrics = evaluate_detectrl_predictions(
        labels,
        predictions,
        ids=ids if all(value is not None for value in ids) else None,
        parse_sources=parse_sources,
        strict_formats=strict_formats,
    )
    elapsed_seconds = time.time() - started
    metrics["runtime"] = {
        "elapsed_seconds_this_session": elapsed_seconds,
        "max_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
            if device.type == "cuda"
            else None
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "result.json").write_text(
        json.dumps(
            {"preds": predictions, "labels": labels, "ids": ids},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_confusion_csv(
        output_dir / "confusion_matrix.csv",
        metrics["test_overall"]["confusion_matrix"],
    )
    _write_report(output_dir / "REPORT.md", metrics, config, elapsed_seconds)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
