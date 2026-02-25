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
#     "trl",
#     "math-verify",
#     "latex2sympy2_extended",
# ]
# ///

"""
MaxRL (Maximum Likelihood Reinforcement Learning) training script.

MaxRL is a variant of GRPO that normalizes advantages by the group mean reward
instead of the group standard deviation. For binary rewards this is equivalent
to dividing by the empirical success rate, giving larger updates on harder
problems where the model rarely succeeds.

Reference: https://arxiv.org/abs/2602.02710

# Full training on a math reasoning dataset
```
python examples/scripts/maxrl.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir maxrl-Qwen2.5-1.5B-Instruct \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_steps 500 \
    --report_to wandb
```

# With vLLM for faster generation
```
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/maxrl.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --output_dir maxrl-Qwen2.5-7B-Instruct \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --use_vllm \
    --vllm_mode colocate \
    --max_steps 1000 \
    --report_to wandb
```
"""

import re

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The assistant solves math problems step by step, "
    "then gives the final answer inside \\boxed{}."
)


def extract_boxed(text: str) -> str | None:
    """Return the last \\boxed{...} content from text, or None if absent."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def get_completion_text(completion) -> str:
    """Return plain text from a completion, whether it is a string or a list of message dicts."""
    if isinstance(completion, str):
        return completion
    # Conversational format: list of {"role": ..., "content": ...} dicts.
    # Use the last assistant turn's content.
    for msg in reversed(completion):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def accuracy_reward(completions, answer, **kwargs):
    """Binary reward: 1.0 if the extracted answer matches the reference, else 0.0."""
    rewards = []
    for completion, ref in zip(completions, answer, strict=False):
        pred = extract_boxed(get_completion_text(completion))
        rewards.append(1.0 if pred is not None and pred == ref.strip() else 0.0)
    return rewards


def format_example(example):
    """Format a dataset row into the conversational prompt format expected by GRPOTrainer."""
    problem = example.get("problem") or example.get("question") or example.get("input", "")
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return {"prompt": prompt, "answer": example.get("answer") or example.get("solution", "")}


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # MaxRL uses scale_rewards="mean" — set it here so the example works
    # out of the box even if the flag is not passed on the command line.
    if training_args.scale_rewards != "mean":
        training_args.scale_rewards = "mean"

    model_name = model_args.model_name_or_path or "Qwen/Qwen2.5-1.5B-Instruct"

    # Load a math reasoning dataset. NuminaMath-CoT is a good default for math RL.
    dataset_name = script_args.dataset_name if script_args.dataset_name else "AI-MO/NuminaMath-CoT"
    all_splits = load_dataset(dataset_name)
    raw_train = all_splits["train"]
    dataset = raw_train.map(format_example, remove_columns=raw_train.column_names)

    # Optional: hold out a small eval split
    if "test" in all_splits:
        raw_eval = all_splits["test"]
        eval_dataset = raw_eval.map(format_example, remove_columns=raw_eval.column_names)
    else:
        eval_dataset = None

    trainer = GRPOTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        reward_funcs=accuracy_reward,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
