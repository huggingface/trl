# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GOLD VLM distillation on GEOQA_R1V.

# Same-family distillation (Qwen3-VL-8B → Qwen3-VL-2B)
# Uses JSD loss. Same architecture and tokenizer, so standard distillation works directly.
# vLLM enabled for faster on-policy generation.
accelerate launch examples/scripts/gold_vlm.py \
    --student_model_name Qwen/Qwen3-VL-2B-Instruct \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct

# Cross-family distillation (Qwen3-VL-8B → LFM2.5-VL-1.6B)
# Uses ULD loss for different tokenizers/processors. vLLM is disabled because this path uses local VLM generation.
accelerate launch examples/scripts/gold_vlm.py \
    --student_model_name LiquidAI/LFM2.5-VL-1.6B \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct \
    --use_uld_loss \
    --no-use_vllm
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from trl.experimental.gold import GOLDConfig, GOLDTrainer


SYSTEM_PROMPT = "Answer with a single number followed by the ° symbol."


def normalize_solution(solution):
    solution = str(solution).replace("<answer>", "").replace("</answer>", "").strip()
    if solution and not solution.endswith("°"):
        solution = f"{solution}°"
    return solution


def make_conversation(example):
    """Convert GEOQA_R1V row into the chat format expected by TRL VLM trainers."""
    return {
        "prompt": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["problem"]},
                ],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": normalize_solution(example["solution"])}],
            },
        ],
        "image": example["image"],
    }


def filter_big_images(example):
    image = example["image"]
    return image.size[0] < 512 and image.size[1] < 512


def convert_to_rgb(example):
    image = example["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["image"] = image
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--teacher_model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lmbda", type=float, default=0.5)
    parser.add_argument("--use_uld_loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm_mode", type=str, default="colocate")
    cli_args = parser.parse_args()

    # ──────────────────────────────────────────────
    # Models
    # ──────────────────────────────────────────────
    student_model = AutoModelForImageTextToText.from_pretrained(
        cli_args.student_model_name, torch_dtype=torch.bfloat16
    )
    teacher_model = AutoModelForImageTextToText.from_pretrained(
        cli_args.teacher_model_name, torch_dtype=torch.bfloat16
    )

    # Freeze everything except the language model head
    for name, param in student_model.named_parameters():
        if "language_model" not in name:
            param.requires_grad = False

    processor = AutoProcessor.from_pretrained(cli_args.student_model_name, padding_side="left")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=r"^.*language_model.*\.(q_proj|k_proj|v_proj)$",
    )

    # ──────────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────────
    dataset = load_dataset("leonardPKU/GEOQA_R1V_Train_8K", split="train")
    dataset = dataset.filter(filter_big_images)
    dataset = dataset.map(convert_to_rgb)
    dataset = dataset.map(make_conversation)

    # Hold out 5% for evaluation
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # ──────────────────────────────────────────────
    # Training config
    # ──────────────────────────────────────────────
    loss_name = "uld" if cli_args.use_uld_loss else "jsd"
    student_short = cli_args.student_model_name.split("/")[-1]
    teacher_short = cli_args.teacher_model_name.split("/")[-1]
    run_name = f"gold-{student_short}-from-{teacher_short}-{loss_name}"

    args = GOLDConfig(
        output_dir=run_name,
        run_name=run_name,
        # GOLD-specific
        lmbda=cli_args.lmbda,
        beta=0.5,
        temperature=0.6,
        max_completion_length=128,
        max_grad_norm=1.0,
        teacher_model_name_or_path=cli_args.teacher_model_name,
        num_generations=1,
        use_uld_loss=cli_args.use_uld_loss,
        uld_crossentropy_weight=0.5,
        uld_distillation_weight=0.5,
        # vLLM
        use_vllm=cli_args.use_vllm,
        vllm_mode=cli_args.vllm_mode,
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=1024,
        max_length=2048,
        # Training schedule
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=1e-4,
        warmup_steps=10,
        # Evaluation
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=25,
        # Precision
        bf16=True,
        # Logging
        logging_steps=10,
        log_completions=True,
        report_to="wandb",
    )

    # ──────────────────────────────────────────────
    # Trainer
    # ──────────────────────────────────────────────
    trainer = GOLDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
