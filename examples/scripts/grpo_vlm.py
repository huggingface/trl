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

"""
pip install math_verify

# Tested on 8x H100 GPUs
accelerate launch \
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --dataset_name lmms-lab/multimodal-open-r1-8k-verified \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --per_device_train_batch_size 8 \
    --output_dir grpo-Qwen2-VL-2B-Instruct \
    --bf16 true \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj"
"""

import re

import torch
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers import AutoProcessor, AutoModelForImageTextToText

from trl.trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.remove_unused_columns = False
    training_args.max_prompt_length = None  # needs the full length to not cut off image tokens

    ################
    # Model & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="train[:5%]")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    def make_conversation(example):
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["problem"]},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "image": example["image"],
        }

    train_dataset = split_dataset[script_args.dataset_train_split]
    eval_dataset = split_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    # Convert the dataset to the required format
    train_dataset = train_dataset.map(make_conversation)
    train_dataset = train_dataset.remove_columns(["problem", "original_question", "original_answer"])
    if eval_dataset:
        eval_dataset = eval_dataset.map(make_conversation)
        eval_dataset = eval_dataset.remove_columns(["problem", "original_question", "original_answer"])

    ################
    # Reward Function for Training
    ################
    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        rewards = [1.0 if match else 0.0 for match in matches]
        return rewards

    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        rewards = []

        for completion, sol in zip(completions, solution):
            try:
                gold_parsed = parse(sol, extraction_mode="first_match")
            except Exception:
                gold_parsed = []

            if len(gold_parsed) != 0:
                # Try parsing predicted answer too
                try:
                    answer_parsed = parse(
                        completion,
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
                    print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                    reward = None
            else:
                # fallback to text match
                reward = float(completion.strip().lower() == sol.strip().lower())

            rewards.append(reward)

        return rewards

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
