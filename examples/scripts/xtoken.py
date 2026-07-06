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
# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
# X-Token off-policy distillation with GOLDTrainer:
python examples/scripts/xtoken.py \
    --student-model meta-llama/Llama-3.2-1B-Instruct \
    --teacher-model Qwen/Qwen3-4B \
    --projection-matrix cross_tokenizer_data/projection_map_..._top_4_sorted.pt \
    --loss-type p_kl \
    --max-steps 100

# With Nemotron text corpus (matches the NeMo-RL reference run):
python examples/scripts/xtoken.py \
    --student-model meta-llama/Llama-3.2-1B-Instruct \
    --teacher-model Qwen/Qwen3-4B \
    --projection-matrix cross_tokenizer_data/projection_map_..._top_4_sorted.pt \
    --dataset nemotron \
    --max-length 512

Build the projection matrix first with the scripts in trl/experimental/gold/scripts/xtoken/. See
https://huggingface.co/papers/2605.21699.
"""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.gold import GOLDConfig, GOLDTrainer


# ~4 chars/token; keep samples comfortably within max_length
_NEMOTRON_CHARS_PER_SAMPLE = 1800


def build_chatbot_arena_dataset(split="train", num_samples=2000):
    return load_dataset("trl-lib/chatbot_arena_completions", split=f"{split}[:{num_samples}]")


def build_nemotron_dataset(num_samples=2000):
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/nvidia/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Formal-Logic/*.parquet",
        split="train",
    )
    ds = ds.select(range(min(num_samples, len(ds))))

    def to_messages(example):
        text = example["text"][:_NEMOTRON_CHARS_PER_SAMPLE]
        return {"messages": [{"role": "user", "content": "Continue:"}, {"role": "assistant", "content": text}]}

    return ds.map(to_messages, remove_columns=ds.column_names)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--student-model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--teacher-model", default="Qwen/Qwen3-4B")
    p.add_argument("--projection-matrix", required=True)
    p.add_argument("--dataset", default="nemotron", choices=["chatbot_arena", "nemotron"])
    p.add_argument("--num-samples", type=int, default=2000)
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
    p.add_argument("--xtoken-ce-scale", type=float, default=0.1)
    p.add_argument("--xtoken-vocab-topk", type=int, default=8192)
    p.add_argument("--xtoken-dynamic-scaling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", default="output/xtoken_run")
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--report-to", default="trackio")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.dataset == "nemotron":
        train_dataset = build_nemotron_dataset(num_samples=args.num_samples)
    else:
        train_dataset = build_chatbot_arena_dataset(num_samples=args.num_samples)

    config = GOLDConfig(
        # Models
        teacher_model_init_kwargs={"dtype": "bfloat16" if args.bf16 else "auto"},
        teacher_tokenizer_name_or_path=args.teacher_model,
        # Sequence lengths
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        lmbda=0.0,
        beta=1.0,
        # X-Token config
        xtoken_loss_type=args.loss_type,
        xtoken_projection_matrix_path=args.projection_matrix,
        xtoken_temperature=args.xtoken_temperature,
        xtoken_kl_weight=args.xtoken_kl_weight,
        xtoken_ce_scale=args.xtoken_ce_scale,
        xtoken_vocab_topk=args.xtoken_vocab_topk,
        xtoken_dynamic_scaling=args.xtoken_dynamic_scaling,
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
        report_to=args.report_to,
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
