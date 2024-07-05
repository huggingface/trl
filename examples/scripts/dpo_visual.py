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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --dataset_num_proc 32 \
    --output_dir dpo_idefics_rlaif-v \
    --bf16 \
    --torch_dtype bfloat16 \
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
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path, do_image_splitting=False)
    tokenizer = processor.tokenizer
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
        # The prompt can be either a string or a list. In some datasets, the prompt is just a common string
        # for both rejected and chosen (already included in chosen and rejected) and is not meant to be used
        # separately. In other datasets, the prompt is intended to be used as a prefix for rejected and chosen,
        # and in such cases, it is properly formatted as a list with keys "role" and "content".
        # Example 1:
        # row = {"prompt": "What does detox mean?",
        #        "chosen": [{"content": "What does detox mean?", "role": "user"}, {"content": "It means to get rid of the toxins.", "role": "assistant"}],
        #        "rejected": [{"content": "What does detox mean?", "role": "assistant"}, {"content": "I don't know.", "role": "user"}]}
        # Example 2:
        # row = {"prompt": [{"content": "What does detox mean?", "role": "user"}],
        #        "chosen": [{"content": "It means to get rid of the toxins.", "role": "assistant"}],
        #        "rejected": [{"content": "I don't know.", "role": "user"}]}
        if "prompt" in row and isinstance(row["prompt"], list):
            row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False)

        row["chosen"] = processor.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = processor.apply_chat_template(row["rejected"], tokenize=False)

        if "images" in row:
            for img in row["images"]:  # Resize each image so the largest side is 640 pixels
                img.thumbnail((640, 640))  # Resize the image to at most 640x640 pixels
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
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()
    trainer.push_to_hub
    with save_context:
        trainer.save_model(training_args.output_dir)
