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
GOLD VLM distillation on MMK12.

# Example 1 — Same-family distillation (SmolVLM-500M → SmolVLM-256M)
# Uses JSD loss. Same architecture and tokenizer, so standard distillation works directly.
# vLLM enabled for faster on-policy generation.
accelerate launch examples/scripts/gold_vlm.py \
    --student_model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --teacher_model_name HuggingFaceTB/SmolVLM-500M-Instruct \
    --lmbda 0.5 \
    --use_vllm \
    --vllm_mode colocate

# Example 2 — Cross-family distillation (Qwen2.5-VL-3B → SmolVLM-256M)
# Different architectures have incompatible tokenizers and image token formats,
# so ULD (Universal Logit Distillation) loss is required to align logits across vocabularies.
accelerate launch examples/scripts/gold_vlm.py \
    --student_model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --teacher_model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --use_uld_loss \
    --lmbda 0.0
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from trl.experimental.gold import GOLDConfig, GOLDTrainer


SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide the user with the answer. "
    "Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
)


def make_conversation(example):
    """Convert MMK12 row into the chat format expected by TRL VLM trainers."""
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
                    {"type": "text", "text": example["question"]},
                ],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(example["answer"])}],
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
    parser.add_argument("--student_model_name", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--teacher_model_name", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--use_uld_loss", action="store_true")
    parser.add_argument("--lmbda", type=float, default=0.5)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate")
    cli_args = parser.parse_args()

    # ──────────────────────────────────────────────
    # Models
    # ──────────────────────────────────────────────
    student_model = AutoModelForImageTextToText.from_pretrained(cli_args.student_model_name, dtype=torch.bfloat16)
    teacher_model = AutoModelForImageTextToText.from_pretrained(cli_args.teacher_model_name, dtype=torch.bfloat16)

    # Freeze everything except the language model head
    for name, param in student_model.named_parameters():
        if "language_model" not in name:
            param.requires_grad = False

    processor = AutoProcessor.from_pretrained(cli_args.student_model_name, padding_side="left")

    # toy example to fit small GPUs
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj"],
    )

    # ──────────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────────
    dataset = load_dataset("FanqingM/MMK12", split="train[:5%]")
    dataset = dataset.filter(filter_big_images)
    dataset = dataset.map(convert_to_rgb)
    dataset = dataset.map(make_conversation)

    # ──────────────────────────────────────────────
    # Training config
    # ──────────────────────────────────────────────
    args = GOLDConfig(
        output_dir="gold-vlm-distillation",
        # GOLD-specific
        lmbda=cli_args.lmbda,
        beta=0.5,
        temperature=0.9,
        max_completion_length=256,
        teacher_model_name_or_path=cli_args.teacher_model_name,
        num_generations=1,
        use_uld_loss=cli_args.use_uld_loss,
        # vLLM
        use_vllm=cli_args.use_vllm,
        vllm_mode=cli_args.vllm_mode,
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=8192,
        # VLM image tokens expand during processing, so the default max_length (1024) is often too small.
        # Which will lead to shifted_student_logits become an empty Tensor.
        max_length=2048,
        # Training schedule
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-5,
        warmup_steps=10,
        # Precision
        bf16=True,
        # Logging
        logging_steps=1,
        log_completions=True,
        report_to="none",
    )

    # ──────────────────────────────────────────────
    # Trainer
    # ──────────────────────────────────────────────
    trainer = GOLDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
