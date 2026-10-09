#!/usr/bin/env python3
"""Run a small, resumable GTA training experiment on DetectRL-ZH.

Unlike ``hf_detectrlzh_one_step.py``, this script samples completion groups
from the current policy.  It is intentionally small enough for a Colab
feasibility run and writes every step plus recovery checkpoints to a supplied
output directory (normally a Google Drive path).
"""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gta_repro.detectrl_zh import (
    DETECTRL_LABELS,
    HUMAN_LABEL,
    MACHINE_LABEL,
    build_detectrl_prompt,
    label_to_text,
    load_detectrl_json,
)
from gta_repro.rewards import score_completion
from gta_repro.trainer import gta_optimizer_step


def _balanced_subset(records, size, seed):
    if size < 2 or size % 2:
        raise ValueError("--num-samples must be an even integer >= 2")
    human = [record for record in records if record.label == HUMAN_LABEL]
    machine = [record for record in records if record.label == MACHINE_LABEL]
    per_class = size // 2
    if len(human) < per_class or len(machine) < per_class:
        raise ValueError("requested subset is larger than one of the two classes")
    rng = random.Random(seed)
    subset = rng.sample(human, per_class) + rng.sample(machine, per_class)
    rng.shuffle(subset)
    return subset


def _generate_groups(
    model,
    tokenizer,
    prompts,
    *,
    group_size,
    max_length,
    max_new_tokens,
    temperature,
    top_p,
):
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length - max_new_tokens,
        return_tensors="pt",
    )
    encoded = {key: value.to(next(model.parameters()).device) for key, value in encoded.items()}
    input_width = encoded["input_ids"].shape[1]
    model.eval()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            num_return_sequences=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    model.train()
    tokenizer.padding_side = previous_padding_side
    decoded = tokenizer.batch_decode(
        generated[:, input_width:], skip_special_tokens=True
    )
    return [
        [decoded[row * group_size + column].strip() for column in range(group_size)]
        for row in range(len(prompts))
    ]


def _load_model(args, dtype, device):
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.resume_from:
        model = PeftModel.from_pretrained(
            base, str(Path(args.resume_from) / "adapter"), is_trainable=True
        )
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    model.to(device)
    model.train()
    return model


def _save_checkpoint(model, optimizer, output_dir, step, state):
    checkpoint = output_dir / "checkpoints" / f"step_{step:04d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint / "adapter")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latest = output_dir / "latest_checkpoint.json"
    latest.write_text(
        json.dumps({"step": step, "path": str(checkpoint)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Resumable DetectRL-ZH GTA micro-training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from")
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args()

    if args.group_size < 2:
        raise ValueError("--group-size must be at least 2")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    if args.num_samples < args.steps * args.batch_size:
        raise ValueError("--num-samples must cover steps * batch-size")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    generations_path = output_dir / "generations.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = _load_model(args, dtype, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    start_step = 0
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        state = json.loads((resume_dir / "trainer_state.json").read_text(encoding="utf-8"))
        start_step = int(state["completed_steps"])
        optimizer.load_state_dict(
            torch.load(resume_dir / "optimizer.pt", map_location=device, weights_only=True)
        )

    reference_model = None
    if not args.no_reference:
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

    records = _balanced_subset(load_detectrl_json(args.data), args.num_samples, args.seed)
    run_config = vars(args).copy()
    run_config.update(
        {
            "dtype": str(dtype),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "phase": "generated_rollout_microtraining",
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for step in range(start_step, args.steps):
        batch = records[step * args.batch_size : (step + 1) * args.batch_size]
        prompts = [build_detectrl_prompt(record.text[: args.max_chars], tokenizer) for record in batch]
        gold_labels = [label_to_text(record.label) for record in batch]
        completion_groups = _generate_groups(
            model,
            tokenizer,
            prompts,
            group_size=args.group_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        scored = [
            [score_completion(completion, gold, DETECTRL_LABELS) for completion in group]
            for group, gold in zip(completion_groups, gold_labels)
        ]
        flat_scores = [score for group in scored for score in group]
        stats = gta_optimizer_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            prompts=prompts,
            completion_groups=completion_groups,
            gold_labels=gold_labels,
            valid_labels=DETECTRL_LABELS,
            reference_model=reference_model,
            max_length=args.max_length,
            beta=0.01,
        )
        stats.update(
            {
                "step": step + 1,
                "format_rate": sum(score.format_reward for score in flat_scores) / len(flat_scores),
                "accuracy_reward_rate": sum(score.accuracy_reward for score in flat_scores) / len(flat_scores),
                "distinct_reward_groups": sum(
                    len({score.total for score in group}) > 1 for group in scored
                ),
                "max_gpu_memory_gb": (
                    round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
                    if device.type == "cuda"
                    else None
                ),
            }
        )
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stats, ensure_ascii=False) + "\n")
        generation_record = {
            "step": step + 1,
            "sample_ids": [record.sample_id for record in batch],
            "gold_labels": gold_labels,
            "completion_groups": completion_groups,
            "rewards": [[score.total for score in group] for group in scored],
        }
        with generations_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(generation_record, ensure_ascii=False) + "\n")

        state = {
            "completed_steps": step + 1,
            "num_samples_consumed": (step + 1) * args.batch_size,
            "last_metrics": stats,
        }
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            checkpoint = _save_checkpoint(model, optimizer, output_dir, step + 1, state)
            stats["checkpoint"] = str(checkpoint)
        print(json.dumps(stats, ensure_ascii=False), flush=True)

    final = {
        "status": "completed",
        "completed_steps": args.steps,
        "metrics_path": str(metrics_path),
        "generations_path": str(generations_path),
        "max_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
            if device.type == "cuda"
            else None
        ),
    }
    (output_dir / "final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
