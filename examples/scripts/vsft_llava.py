# flake8: noqa
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# regular:
python examples/scripts/vsft_llava.py \
    --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft" \
    --model_name_or_path="llava-hf/llava-1.5-7b-hf" \
    --report_to="wandb" \
    --learning_rate=1.4e-5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --output_dir="data/vsft-llava-1.5-7b-hf" \
    --logging_steps=5 \
    --num_train_epochs=1 \
    --push_to_hub \
    --gradient_checkpointing \
    --remove_unused_columns=False \
    --torch_dtype=float16 \
    --fp16=True
    
# peft:
python examples/scripts/vsft_llava.py \
    --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft" \    
    --model_name_or_path="llava-hf/llava-1.5-7b-hf" \
    --report_to="wandb" \
    --learning_rate=1.4e-5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --output_dir="data/vsft-llava-1.5-7b-hf" \
    --logging_steps=5 \
    --num_train_epochs=1 \
    --push_to_hub \
    --gradient_checkpointing \
    --remove_unused_columns=False \
    --torch_dtype=float16 \
    --fp16=True \ 
    --use_peft=True \
    --lora_r=64 \
    --lora_alpha=16 \
    --lora_target_modules=all-linear"

# evaluation:
 
To evaluate, first install the lmms-eval framework: pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
then run:
accelerate launch --num_processes=8 -m lmms_eval \
        --model llava_hf \
        --model_args pretrained=llava-hf/llava-1.5-7b-hf \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_sample    
"""

import logging
import os
from contextlib import nullcontext

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from trl.env_utils import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model, Tokenizer & Processor
    ################
    LLAVA_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)
    processor.tokenizer = tokenizer

    model = LlavaForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    ################
    # Create a data collator to encode text and image pairs
    ################

    class LLavaDataCollator:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, examples):
            texts = []
            images = []
            for example in examples:
                if len(example["images"]) > 1:
                    raise ValueError("This collator only supports one image per example")
                messages = example["messages"]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
                images.append(example["images"][0])

            batch = self.processor(texts, images, return_tensors="pt", padding=True)

            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            return batch

    data_collator = LLavaDataCollator(processor)

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(sft_script_args.dataset_name)
    train_dataset = raw_datasets[sft_script_args.dataset_train_split]
    eval_dataset = raw_datasets[sft_script_args.dataset_test_split]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",  # need a dummy field
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=data_collator,
            dataset_kwargs={"skip_prepare_dataset": True},
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
        trainer.push_to_hub()
        if Accelerator().is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
