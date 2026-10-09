#!/usr/bin/env python3
"""Run one offline GTA update for Chinese Human/Machine detection."""

import argparse
import json

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gta_repro.detectrl_zh import (
    DETECTRL_LABELS,
    HUMAN_LABEL,
    MACHINE_LABEL,
    build_detectrl_prompt,
    label_to_text,
    load_detectrl_json,
)
from gta_repro.trainer import gta_optimizer_step


def _smoke_records(path, max_chars):
    if path:
        records = load_detectrl_json(path)
        selected = []
        for label in (HUMAN_LABEL, MACHINE_LABEL):
            selected.append(next(record for record in records if record.label == label))
        return [(record.text[:max_chars], record.label) for record in selected]
    return [
        ("这家小店我去了三次，味道不算惊艳，但老板记得我不吃香菜。", HUMAN_LABEL),
        (
            "人工智能技术正在快速发展，并在教育、医疗和工业领域展现出广泛的应用潜力。",
            MACHINE_LABEL,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="DetectRL-ZH GTA offline one-step smoke")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", help="official train.json; selects one sample per class")
    parser.add_argument("--max-chars", type=int, default=160)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_source = args.model
    if args.local_files_only:
        model_source = snapshot_download(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, local_files_only=args.local_files_only
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-5
    )

    records = _smoke_records(args.data, args.max_chars)
    prompts = [build_detectrl_prompt(text, tokenizer) for text, _ in records]
    gold_labels = [label_to_text(label) for _, label in records]
    completion_groups = []
    for gold in gold_labels:
        wrong = "机器生成" if gold == "人类撰写" else "人类撰写"
        completion_groups.append(
            [
                f"<guess>{gold}</guess><think>检查整体语言模式后维持判断。</think><answer>{gold}</answer>",
                f"<guess>{wrong}</guess><think>这是同组中的错误候选。</think><answer>{wrong}</answer>",
            ]
        )
    stats = gta_optimizer_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        prompts=prompts,
        completion_groups=completion_groups,
        gold_labels=gold_labels,
        valid_labels=DETECTRL_LABELS,
        max_length=768,
        beta=0.01,
    )
    stats.update(
        {
            "task": "DetectRL-ZH document-level Human/Machine classification",
            "labels": {"0": "人类撰写", "1": "机器生成"},
            "n_prompts": len(prompts),
            "group_size": 2,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        }
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

