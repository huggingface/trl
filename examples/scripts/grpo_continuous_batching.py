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
#     "trl[peft]",
#     "datasets",
#     "flash-attn",
# ]
# ///

"""
GRPO training on GSM8K using transformers' continuous batching engine instead of the default
generate(). Continuous batching removes finished sequences from the batch immediately rather than
waiting for the slowest one, which yields faster generation and lower VRAM than default generate()
for tasks with variable completion lengths (N≥32).

The key config parameter is `max_memory_percent` in `transformers_continuous_batching_config`,
which caps the paged KV cache as a fraction of free VRAM. TRL defaults to 0.5; tune it down to
0.3-0.4 for large generation batches to leave room for the training backward pass.

python examples/scripts/grpo_continuous_batching.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --output_dir grpo-gsm8k-continuous-batching \
    --num_generations 32 \
    --max_completion_length 1024 \
    --use_peft \
    --log_completions

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_continuous_batching.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --output_dir grpo-gsm8k-continuous-batching \
    --num_generations 32 \
    --max_completion_length 1024 \
    --use_peft \
    --log_completions
"""

import logging
import re

import torch
from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


logging.getLogger("ContinuousBatchingLogger").setLevel(logging.ERROR)

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the problem step by step, then provide the final "
    "numeric answer on the last line in the format: #### <number>"
)


def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\d,]+)", text)
    return match.group(1).replace(",", "") if match else None


def correctness_reward(completions, reference_answer, **kwargs):
    rewards = []
    for completion, ref in zip(completions, reference_answer):
        predicted = extract_answer(completion if isinstance(completion, str) else completion[-1]["content"])
        rewards.append(1.0 if predicted is not None and predicted == ref else 0.0)
    return rewards


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name or "openai/gsm8k",
        script_args.dataset_config or "main",
        split=script_args.dataset_train_split,
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "reference_answer": example["answer"].split("####")[-1].strip().replace(",", ""),
        }

    dataset = dataset.map(make_conversation, remove_columns=dataset.column_names)

    ################
    # Training
    ################
    training_args.use_transformers_continuous_batching = True
    if training_args.transformers_continuous_batching_config is None:
        training_args.transformers_continuous_batching_config = {"use_cuda_graph": False}

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=correctness_reward,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
