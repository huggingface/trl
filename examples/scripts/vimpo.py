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
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
NuminaMath example: VIMPO on math reasoning with verifiable rewards.

pip install math_verify latex2sympy2_extended peft trackio

# For Qwen/Qwen3-0.6B
pip install num2words==0.5.14

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/vimpo.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --output_dir vimpo-Qwen3-0.6B \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 8 \
    --vimpo_beta 5e-4 \
    --vimpo_actor_coeff 5e-3
"""

import torch
from datasets import load_dataset

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.vimpo import VIMPOConfig, VIMPOTrainer
from trl.rewards import accuracy_reward, think_format_reward


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, VIMPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Dataset
    ################
    train_dataset, eval_dataset = load_dataset("AI-MO/NuminaMath-TIR", split=["train[:5%]", "test[:5%]"])

    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the answer. The reasoning "
        "process must be enclosed within <think></think> tags, and the final answer must be enclosed in \\boxed{}, "
        "i.e., <think>\nThis is my reasoning.\n</think>\n\\boxed{answer}"
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)

    train_dataset = train_dataset.remove_columns(["messages", "problem"])
    eval_dataset = eval_dataset.remove_columns(["messages", "problem"])

    ################
    # Training
    ################
    trainer = VIMPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or "AI-MO/NuminaMath-TIR")
