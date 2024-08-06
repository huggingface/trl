# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
accelerate launch examples/scripts/dpo_visual.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --model_name_or_path HuggingFaceM4/idefics2-8b \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir dpo_idefics_rlaif-v \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules=all-linear
"""

import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from accelerate import PartialState

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)

    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForVision2Seq.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        do_image_splitting=False,
    )
    tokenizer = processor.tokenizer

    # Set up the chat template
    if model.config.model_type == "idefics2":
        pass  # the processor already has a valid chat template
    elif model.config.model_type == "paligemma":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] }}<|im_end|>{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""
    elif model.config.model_type == "llava":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    def process(row):
        row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False)
        row["chosen"] = processor.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = processor.apply_chat_template(row["rejected"], tokenize=False)
        return row

    with PartialState().local_main_process_first():
        ds = ds.map(process, num_proc=training_args.dataset_num_proc)
    train_dataset = ds[args.dataset_train_split]
    eval_dataset = ds[args.dataset_test_split]

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
