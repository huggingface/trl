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
Full training:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048

LoRA:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 /
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import warnings

import torch
from accelerate import PartialState
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
from trl.commands.cli_utils import RewardScriptArguments
from trl.extras.dataset_formatting import conversations_formatting_function


tqdm.pandas()


if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

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
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
        )

    #############################
    # Load and preprocess dataset
    #############################
    raw_datasets = load_dataset(args.dataset_name)

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the chosen/rejected columns are in the OpenAI messages format.
        chosen_fn = conversations_formatting_function(tokenizer, "chosen")
        rejected_fn = conversations_formatting_function(tokenizer, "rejected")
        raw_datasets = raw_datasets.map(
            lambda x: {"chosen": chosen_fn(x), "rejected": rejected_fn(x)}, num_proc=config.dataset_num_proc
        )
        # Tokenize inputs
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=config.dataset_num_proc,
        )
        # Filter out examples that are too long
        raw_datasets = raw_datasets.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.max_length
            and len(x["input_ids_rejected"]) <= config.max_length,
            num_proc=config.dataset_num_proc,
        )

    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(config.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model(config.output_dir)
    trainer.push_to_hub()
