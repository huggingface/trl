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
#     "peft",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
# Recommended small student: LFM2.5-1.2B
python examples/scripts/gold.py \
    --model_name_or_path LiquidAI/LFM2.5-1.2B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen3-4B \
    --dataset_name nvidia/Nemotron-Pretraining-Specialized-v1.1 \
    --dataset_config Nemotron-Pretraining-Formal-Logic \
    --use_uld_loss \
    --lmbda 0.0 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-lfm2.5-1.2b \
    --num_train_epochs 1 \
    --push_to_hub

# Recommended Gemma 4 student. Gemma 4 uses SentencePiece, so cross-tokenizer
# ULD requires the positional alignment path (`use_extended_uld=False`). This
# script uses Gemma 4 in text-only mode; use gold_vlm.py for multimodal training.
python examples/scripts/gold.py \
    --model_name_or_path google/gemma-4-E4B-it \
    --teacher_model_name_or_path Qwen/Qwen3-4B \
    --dataset_name nvidia/Nemotron-Pretraining-Specialized-v1.1 \
    --dataset_config Nemotron-Pretraining-Formal-Logic \
    --use_uld_loss \
    --no_use_extended_uld \
    --lmbda 0.0 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-gemma-4-e4b \
    --num_train_epochs 1 \
    --push_to_hub \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 16
"""

import logging

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold.gold_config import GOLDConfig
from trl.experimental.gold.gold_trainer import GOLDTrainer


logger = logging.getLogger(__name__)


def split_text_for_gold(example, tokenizer, max_length):
    """Split a raw language-modeling example into a 75% prompt and 25% completion."""
    input_ids = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
    if max_length is not None:
        input_ids = input_ids[-max_length:]
    split_idx = max(1, int(len(input_ids) * 0.75))
    return {
        "prompt": tokenizer.decode(input_ids[:split_idx]),
        "completion": tokenizer.decode(input_ids[split_idx:]),
    }


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    if training_args.teacher_tokenizer_name_or_path is None and training_args.use_uld_loss:
        training_args.teacher_tokenizer_name_or_path = training_args.teacher_model_name_or_path
    teacher_model_kwargs = dict(
        revision=training_args.teacher_model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=True,
        quantization_config=quantization_config,
    )
    if training_args.teacher_model_init_kwargs is not None:
        teacher_model_kwargs.update(training_args.teacher_model_init_kwargs)
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset_sample = next(iter(dataset[script_args.dataset_train_split]))
    if "text" in dataset_sample and "prompt" not in dataset_sample and "messages" not in dataset_sample:
        dataset = dataset.map(
            split_text_for_gold,
            fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.max_length},
            num_proc=training_args.dataset_num_proc,
            desc="Splitting raw text into GOLD prompts and completions",
        )
        dataset = dataset.filter(
            lambda example: bool(example["prompt"] and example["completion"]),
            num_proc=training_args.dataset_num_proc,
            desc="Removing examples that are too short for GOLD",
        )

    ################
    # Training
    ################
    eval_dataset = None
    if training_args.eval_strategy != "no":
        if script_args.dataset_test_split in dataset:
            eval_dataset = dataset[script_args.dataset_test_split]
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
        elif "dev" in dataset:
            eval_dataset = dataset["dev"]

    trainer = GOLDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
