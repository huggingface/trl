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
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "torchvision",
#     "kernels",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen2.5-VL-3B-Instruct
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/online_dpo_vlm.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --reward_model_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir online-dpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --max_length 1536 \
    --max_new_tokens 1024 \
    --use_vllm \
    --vllm_mode server \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2

# For HuggingFaceTB/SmolVLM2-2.2B-Instruct
pip install num2words

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/online_dpo_vlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --reward_model_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --output_dir online-dpo-SmolVLM2-2.2B-Instruct \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_length 1536 \
    --max_new_tokens 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2

# Single GPU test command:
python examples/scripts/online_dpo_vlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --reward_model_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --output_dir online-dpo-SmolVLM2-2.2B-Instruct-test \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_length 1536 \
    --max_new_tokens 128 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 2 \
    --logging_steps 1 \
    --trust_remote_code
"""

import os

import torch
import transformers
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers import AutoConfig, AutoProcessor, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load the VLM model using correct architecture (from GRPO pattern)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    architecture = getattr(transformers, config.architectures[0])
    model = architecture.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    # For VLM online DPO, using a reward model is complex because it needs images
    # Instead, we'll use a simple random judge for testing
    # In production, you'd want to use a proper text-only reward model or a custom judge
    reward_model = None
    reward_processor = None

    # Load processor for main model
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset("lmms-lab/multimodal-open-r1-8k-verified", split="train")
    dataset = dataset.train_test_split(test_size=100, seed=42)

    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
        "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
        "reasoning.\n</think>\nThis is my answer."
    )

    def make_conversation(example):
        # Create conversational format that OnlineDPOTrainer expects
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        return {"prompt": prompt, "image": example["image"]}

    dataset = dataset.map(make_conversation)

    # Filter big images (from GRPO pattern)
    def filter_big_images(example):
        image = example["image"]
        return image.size[0] < 512 and image.size[1] < 512

    dataset = dataset.filter(filter_big_images)

    def convert_to_rgb(example):
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    dataset = dataset.map(convert_to_rgb)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    ################
    # Reward Function for Training (same as GRPO VLM)
    ################
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

    ################
    # Training
    ################
    trainer = OnlineDPOTrainer(
        model=model,
        reward_funcs=[think_format_reward, accuracy_reward],  # Use same reward functions as GRPO VLM
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    # Add completion logging callback (from online DPO pattern)
    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name="lmms-lab/multimodal-open-r1-8k-verified")
