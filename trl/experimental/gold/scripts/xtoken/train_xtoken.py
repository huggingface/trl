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
"""X-Token off-policy distillation with GOLDTrainer.

Pure off-policy (lmbda=0): no on-policy generation, the dataset feeds the student and teacher tokenizers directly via
DataCollatorForChatML.

Usage:

    python trl/experimental/gold/scripts/xtoken/train_xtoken.py
        --projection-matrix cross_tokenizer_data/projection_map_..._top_4_sorted.pt --loss-type p_kl --max-steps 100

Build the projection matrix first with the three scripts in this directory. See
https://huggingface.co/papers/2605.21699.
"""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.gold import GOLDConfig, GOLDTrainer


def build_dataset(split="train", num_samples=2000):
    """Load chatbot_arena_completions (the proven GOLD dataset)."""
    return load_dataset("trl-lib/chatbot_arena_completions", split=f"{split}[:{num_samples}]")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--student-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--teacher-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--projection-matrix", required=True)
    p.add_argument("--loss-type", default="p_kl", choices=["p_kl", "h_kl"])
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--xtoken-temperature", type=float, default=1.0)
    p.add_argument("--xtoken-kl-weight", type=float, default=1.0)
    p.add_argument("--xtoken-vocab-topk", type=int, default=32)
    p.add_argument("--output-dir", default="output/xtoken_run")
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--project", default="xtoken-distillation")
    p.add_argument("--trackio-space-id", default="kashif/xtoken-distillation")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = build_dataset()

    config = GOLDConfig(
        # Models
        teacher_model_init_kwargs={"dtype": "bfloat16" if args.bf16 else "auto"},
        teacher_tokenizer_name_or_path=args.teacher_model,
        # Sequence lengths
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        # Pure off-policy: no on-policy generation needed
        lmbda=0.0,
        use_vllm=False,
        # X-Token config
        xtoken_loss_type=args.loss_type,
        xtoken_projection_matrix_path=args.projection_matrix,
        xtoken_temperature=args.xtoken_temperature,
        xtoken_kl_weight=args.xtoken_kl_weight,
        xtoken_vocab_topk=args.xtoken_vocab_topk,
        xtoken_uncommon_topk=8192,
        xtoken_dynamic_scaling=False,
        # Standard KD params (beta=1 → pure KD, no JSD)
        beta=1.0,
        temperature=args.temperature,
        # Training
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.max_steps,
        report_to="trackio",
        project=args.project,
        trackio_space_id=args.trackio_space_id,
        num_generations=1,
    )

    trainer = GOLDTrainer(
        model=args.student_model,
        teacher_model=args.teacher_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
