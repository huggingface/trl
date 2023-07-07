# coding=utf-8
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
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

from trl import RewardTrainer


########################################################################
# This is a fully working simple example to use trl's RewardTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the RewardTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="huggyllama/llama-7b",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="Anthropic/hh-rlhf",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )

    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def create_and_prepare_model(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    # TODO: make it more userfriendly
    device_map = {"": 0}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map=device_map
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # we add `score` to the list of modules to save to
    # correctly save the score head.
    peft_config = LoraConfig(
        lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS", modules_to_save=["score"]
    )

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_and_prepare_dataset(args, tokenizer, num_proc=12):
    dataset = load_dataset(args.dataset_name, split="train[:1%]")
    original_columns = dataset.column_names

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(
                chosen, truncation=True, padding="max_length", max_length=script_args.max_seq_length
            )
            tokenized_rejected = tokenizer(
                rejected, truncation=True, padding="max_length", max_length=script_args.max_seq_length
            )

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_rejected["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    dataset = dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
    return dataset


def main():
    model, tokenizer = create_and_prepare_model(script_args)
    dataset = create_and_prepare_dataset(script_args, tokenizer)

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        learning_rate=script_args.learning_rate,
        optim=script_args.optim,
        max_steps=1,
        lr_scheduler_type=script_args.lr_scheduler_type,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        save_steps=1,
        gradient_checkpointing=script_args.gradient_checkpointing,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_length=script_args.max_seq_length,
    )

    trainer.train()


if __name__ == "__main__":
    main()
