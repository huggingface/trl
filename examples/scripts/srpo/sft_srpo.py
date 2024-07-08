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
import os

os.environ["HUGGINGFACE_CACHE"] = "/workspace/.cache/huggingface"
os.environ["DATA_DIR"] = "./data"
os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HUGGINGFACE_CACHE']}/datasets"
os.environ["HF_HOME"] = f"{os.environ['HUGGINGFACE_CACHE']}/misc"
os.environ["TRANSFORMERS_CACHE"] = f"{os.environ['HUGGINGFACE_CACHE']}/transformers"
os.environ["WANDB_LOG_MODEL"] = "end"

"""
# regular:
python examples/scripts/srpo/sft_srpo.py \
    --model_name_or_path="EleutherAI/pythia-1b-deduped" \
    --report_to="wandb" \
    --learning_rate=2e-4 \
    --per_device_train_batch_size=64 \
    --output_dir="srpo_sft" \
    --logging_steps=1 \
    --num_train_epochs=2 \
    --max_seq_length=512 \
    --max_steps=-1 \
    --eval_steps=20 \
    --lr_scheduler_type="cosine" \
    --warmup_ratio=0.03 \
    --weight_decay=0.1 \
    --dataset_name="trl-internal-testing/tldr-preference-sft-trl-style" \
    --torch_dtype="bfloat16" \
    --add_special_tokens=False \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

from trl.trainer.utils import SIMPLE_SFT_CHAT_TEMPLATE

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
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
        else model_config.torch_dtype
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        # load_in_4bit=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_SFT_CHAT_TEMPLATE

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=1 if config.sanity_check else multiprocessing.cpu_count(),
            load_from_cache_file=not config.sanity_check,
        )

    
    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
