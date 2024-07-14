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
import logging
import multiprocessing
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import SRPOScriptArguments, init_zero_verbose, TrlParser
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE


import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    SRPOConfig,
    SRPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

"""
# post sft:
python examples/scripts/srpo/srpo_tldr.py \
    --model_name_or_path=./srpo_sft_1 \
    --per_device_train_batch_size=32 \
    --learning_rate=2e-5 \
    --gradient_accumulation_steps=4 \
    --logging_steps=10 \
    --eval_steps=100 \
    --eval_strategy="steps" \
    --save_steps=100 \
    --output_dir="srpo_tldr_peft_fix" \
    --logging_first_step \
    --no_remove_unused_columns \
    --report_to=wandb \
    --torch_dtype="bfloat16" \
    --lr_scheduler_type="cosine" \
    --warmup_steps=150 \
    --weight_decay=0.1 \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=32 \
    --num_train_epochs=5 \
    --max_length=700 \
    --max_prompt_length=700 \
    --generate_during_eval=True \
    --attn_implementation="flash_attention_2"\
    --gradient_checkpointing


# regular:
python examples/scripts/srpo/srpo_tldr.py \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --per_device_train_batch_size=32 \
    --learning_rate=2e-5 \
    --gradient_accumulation_steps=4 \
    --logging_steps=10 \
    --eval_steps=100 \
    --eval_strategy="steps" \
    --save_steps=100 \
    --output_dir="srpo_tldr" \
    --logging_first_step \
    --no_remove_unused_columns \
    --report_to=wandb \
    --torch_dtype="bfloat16" \
    --lr_scheduler_type="cosine" \
    --warmup_steps=150 \
    --weight_decay=0.1 \
    --num_train_epochs=5 \
    --max_length=700 \
    --max_prompt_length=700 \
    --generate_during_eval=True \
    --attn_implementation="flash_attention_2"\
    --gradient_checkpointing

# peft:
python examples/scripts/srpo/srpo_tldr.py \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --per_device_train_batch_size=32 \
    --learning_rate=2e-5 \
    --gradient_accumulation_steps=4 \
    --logging_steps=10 \
    --eval_steps=100 \
    --eval_strategy="steps" \
    --save_steps=100 \
    --output_dir="srpo_tldr" \
    --logging_first_step \
    --no_remove_unused_columns \
    --report_to=wandb \
    --torch_dtype="bfloat16" \
    --lr_scheduler_type="cosine" \
    --warmup_steps=150 \
    --weight_decay=0.1 \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=32 \
    --num_train_epochs=5 \
    --max_length=700 \
    --max_prompt_length=700 \
    --generate_during_eval=True \
    --attn_implementation="flash_attention_2"\
    --gradient_checkpointing

"""


if TRL_USE_RICH:
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
    )


if __name__ == "__main__":
    parser = TrlParser((SRPOScriptArguments, SRPOConfig, ModelConfig))
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
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    tokenizer.chat_template = """Below is a reddit POST and the corresponding SUBREDDIT and TITLE{{", and an EXAMPLE SUMMARY." if example else "."}}
Write a both precise and concise summary of the contents of the POST.
{{messages}}
{%- if example %}
EXAMPLE SUMMARY: {{example + "\n"}}
{%- endif %}

TL;DR:
{%- if answer %}
{{answer}}
{%- endif %}
"""
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status("[bold green]Initializing the DPOTrainer...")
    )
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(
            f"[bold green]Training completed! Saving the model to {training_args.output_dir}"
        )
    )

    ################
    # Dataset
    ################
    raw_datasets = load_dataset("trl-internal-testing/tldr-preference-trl-style")
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    # Process to find length of longest inputs upon applying chat template
    def process(row):
        if row["prompt"].endswith("TL;DR:"):
            row["prompt"] = row["prompt"][:-6]
        row["chosen"] = row["chosen"][1]["content"]
        row["rejected"] = row["rejected"][1]["content"]
        # Given SRPO uses example and current answer we choose the longest of
        # chosen and rejected and use that for both to find max length
        if len(row["chosen"]) > len(row["rejected"]):
            longest = len(
                tokenizer.apply_chat_template(
                    row["prompt"], example=row["chosen"], padding=False
                )
            ) + len(row["chosen"])
        else:
            longest = len(
                tokenizer.apply_chat_template(
                    row["prompt"], example=row["rejected"], padding=False
                )
            ) + len(row["rejected"])

        row["longest_length"] = longest

        return row

    train_dataset = train_dataset.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        # load_from_cache_file=False,
    )
    eval_dataset = eval_dataset.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        # load_from_cache_file=False,
    )

    train_dataset = train_dataset.filter(
        lambda x: x["longest_length"] <= training_args.max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["longest_length"] <= training_args.max_length
    ).select(range(1000))
    ################
    # Training
    ################

    training_args.dataset_num_proc = multiprocessing.cpu_count()
    with init_context:
        trainer = SRPOTrainer(
            model,
            model_ref,
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
