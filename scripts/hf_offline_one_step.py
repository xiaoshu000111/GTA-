#!/usr/bin/env python3
"""One real-model GTA update using a locally cached Qwen checkpoint."""

import argparse
import json

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gta_repro.formatting import build_prompt
from gta_repro.trainer import gta_optimizer_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_source = args.model
    if args.local_files_only:
        # Passing the resolved directory also avoids a transformers 4.57.x
        # tokenizer metadata probe that can occur even in offline mode.
        model_source = snapshot_download(args.model, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source, local_files_only=True if args.local_files_only else False
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
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
    )

    labels = ["negative", "positive"]
    prompt = build_prompt("A warm, moving and beautifully acted film.", labels)
    completion_groups = [[
        "<guess>positive</guess><think>The wording is favorable.</think><answer>positive</answer>",
        "<guess>negative</guess><think>I will choose the other label.</think><answer>negative</answer>",
    ]]
    stats = gta_optimizer_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        prompts=[prompt],
        completion_groups=completion_groups,
        gold_labels=["positive"],
        valid_labels=labels,
        max_length=512,
        beta=0.01,
    )
    stats["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
