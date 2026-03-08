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
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Usage:

python examples/scripts/sdpo.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --output_dir sdpo-Qwen3-0.6B \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules q_proj v_proj \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --steps_per_generation 8 \
    --distillation_alpha 1.0 \
    --full_logit_distillation false \
    --sdpo_policy_loss_mode distillation_only

This example uses verifiable math rewards. If your dataset already contains textual environment feedback, pass the
column name via `--feedback_column`; it will be forwarded as `privileged_context` for SDPO reprompting.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer
from trl.rewards import accuracy_reward, think_format_reward


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my reasoning.\n</think>\n"
    "This is my answer."
)


@dataclass
class SDPOScriptArguments(ScriptArguments):
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Optional local dataset path to load with `load_from_disk`. Overrides `dataset_name`."},
    )
    feedback_column: str | None = field(
        default=None,
        metadata={"help": "Optional dataset column containing textual environment feedback to pass as `privileged_context`."},
    )
    eval_num_prompts: int | None = field(
        default=8,
        metadata={"help": "Number of prompts to log during evaluation. Set to 0 to disable completion logging."},
    )


@dataclass
class ExampleSDPOConfig(SDPOConfig):
    scale_rewards: str = field(
        default="group",
        metadata={"help": "Reward normalization mode. Supported: `group`, `batch`, `none`."},
    )


def _make_conversation(example: dict[str, Any], feedback_column: str | None) -> dict[str, Any]:
    prompt = example.get("prompt")
    if prompt is None and "problem" in example:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]

    if prompt is None:
        raise ValueError("Each example must provide either `prompt` or `problem`.")

    output = {"prompt": prompt}

    if "solution" in example:
        output["solution"] = example["solution"]
    elif "answer" in example:
        output["solution"] = example["answer"]

    if feedback_column is not None and feedback_column in example:
        output["privileged_context"] = example[feedback_column]
    elif "privileged_context" in example:
        output["privileged_context"] = example["privileged_context"]

    return output


if __name__ == "__main__":
    parser = TrlParser((SDPOScriptArguments, ExampleSDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    if script_args.dataset_path is not None:
        dataset = load_from_disk(script_args.dataset_path)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    if not isinstance(dataset, DatasetDict):
        raise ValueError("SDPO example expects a dataset with named splits.")

    train_dataset = dataset[script_args.dataset_train_split].map(
        lambda example: _make_conversation(example, script_args.feedback_column),
        remove_columns=dataset[script_args.dataset_train_split].column_names,
    )
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset[script_args.dataset_test_split].map(
            lambda example: _make_conversation(example, script_args.feedback_column),
            remove_columns=dataset[script_args.dataset_test_split].column_names,
        )

    trainer = SDPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    if eval_dataset is not None and script_args.eval_num_prompts:
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        trainer.add_callback(LogCompletionsCallback(trainer, generation_config, num_prompts=script_args.eval_num_prompts))

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or script_args.dataset_path)
