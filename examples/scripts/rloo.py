# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
#     "trl[vllm]",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install math_verify num2words==0.5.14 peft trackio vllm
export TRACKIO_PROJECT="RLOO-NuminaMath-TIR"
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/rloo.py
"""

import os

import torch
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig

from trl import RLOOConfig, RLOOTrainer
from trl.rewards import think_format_reward


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def main():
    # Dataset
    train_dataset, eval_dataset = load_dataset("AI-MO/NuminaMath-TIR", split=["train[:5%]", "test[:5%]"])

    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
        "reasoning.\n</think>\nThis is my answer."
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    train_dataset = train_dataset.map(make_conversation, remove_columns=["messages", "problem"])
    eval_dataset = eval_dataset.map(make_conversation, remove_columns=["messages", "problem"])

    # Reward function for training
    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            try:
                gold_parsed = parse(sol, extraction_mode="first_match")
            except Exception:
                gold_parsed = []

            if len(gold_parsed) != 0:
                # Try parsing predicted answer too
                try:
                    answer_parsed = parse(
                        content,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed="all",
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}, answer: {content}, gold: {sol}")
                    reward = None
            else:
                # fallback to text match
                reward = float(content.strip().lower() == sol.strip().lower())

            rewards.append(reward)

        return rewards

    # Training
    training_args = RLOOConfig(
        output_dir="Qwen3-0.6B-RLOO",
        model_init_kwargs={"dtype": torch.bfloat16},
        learning_rate=1e-5,
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        log_completions=True,
        num_completions_to_print=2,
        max_prompt_length=2048,
        max_completion_length=1024,
        gradient_accumulation_steps=2,
        steps_per_generation=8,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        run_name="Qwen3-0.6B-RLOO-NuminaMath-TIR",
    )

    trainer = RLOOTrainer(
        model="Qwen/Qwen3-0.6B",
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    trainer.push_to_hub(dataset_name="AI-MO/NuminaMath-TIR")


if __name__ == "__main__":
    main()
