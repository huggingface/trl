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
# Full training:
python examples/scripts/gkd.py \
    --model_name_or_path="Qwen/Qwen2-0.5B-Instruct" \
    --teacher_model_name_or_path="Qwen/Qwen2-1.5B-Instruct" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=2e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="gkd_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/gkd.py \
    --model_name_or_path="facebook/opt-125m" \
    --teacher_model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
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

from trl.commands.cli_utils import SFTScriptArguments, TrlParser, init_zero_verbose
from trl.env_utils import strtobool


TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

from datasets import load_dataset
from tqdm.rich import tqdm
from transformers import AutoTokenizer

from trl import (
    GKDConfig,
    GKDTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import pad

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

import torch
from typing import List, Dict, Any, Union
from transformers import DataCollatorForLanguageModeling


class DataCollatorForLastCompletionLM(DataCollatorForLanguageModeling):
    """
    Data collator for language modeling that ignores all tokens except the last completion.
    It also separates prompts and completions for easy access.

    Args:
        tokenizer: The tokenizer to use for encoding the data.
        mlm (bool): Whether to use masked language modeling. Default is False.
        response_template (str): The template that marks the start of a response.
        ignore_index (int): The index to use for ignoring tokens in loss calculation.
    """

    def __init__(
        self, tokenizer, mlm: bool = False, response_template: str = "### Response:\n", ignore_index: int = -100
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = self.tokenizer.encode(response_template, add_special_tokens=False)
        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().torch_call(examples)

        prompts = []
        completions = []

        for i in range(len(examples)):
            input_ids = batch["input_ids"][i]
            labels = batch["labels"][i]

            # Find all occurrences of the response template
            response_starts = [
                j
                for j in range(len(input_ids) - len(self.response_template) + 1)
                if input_ids[j : j + len(self.response_template)].tolist() == self.response_template
            ]

            if not response_starts:
                # If no response template is found, treat the whole input as a prompt
                prompts.append(input_ids)
                completions.append(torch.tensor([]))
                labels[:] = self.ignore_index
            else:
                # Get the start of the last response
                last_response_start = response_starts[-1]

                # Separate prompt and completion
                prompts.append(input_ids[:last_response_start])
                completions.append(input_ids[last_response_start:])

                # Set labels for all tokens before the last response to ignore_index
                labels[:last_response_start] = self.ignore_index

        # Add prompts and completions to the batch
        batch["prompts"] = pad(prompts, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        batch["completions"] = pad(completions, padding_value=self.tokenizer.pad_token_id, padding_side="right")

        return batch


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, GKDConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024

    instruction_template = "### Human:"
    response_template = "### Assistant:"
    collator = DataCollatorForLastCompletionLM(
        # instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

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
        trainer = GKDTrainer(
            model=model_config.model_name_or_path,
            teacher_model=training_args.teacher_model_name_or_path,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=collator,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
