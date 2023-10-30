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

import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from trl import RewardConfig, RewardTrainer


tqdm.pandas()


@dataclass
class ScriptArguments:
    model_name: str = "facebook/opt-350m"
    """the model name"""
    dataset_name: str = "Anthropic/hh-rlhf"
    """the dataset name"""
    dataset_text_field: str = "text"
    """the text field of the dataset"""
    eval_split: str = "none"
    """the dataset split to evaluate on; default to 'none' (no evaluation)"""
    load_in_8bit: bool = False
    """load the model in 8 bits precision"""
    load_in_4bit: bool = False
    """load the model in 4 bits precision"""
    trust_remote_code: bool = True
    """Enable `trust_remote_code`"""
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            output_dir="output",
            per_device_train_batch_size=64,
            num_train_epochs=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            learning_rate=1.41e-5,
            report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=500,
            evaluation_strategy="no",
            max_length=512,
        )
    )
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        ),
    )


args = tyro.cli(ScriptArguments)
args.reward_config.evaluation_strategy = "steps" if args.eval_split != "none" else "no"


# Step 1: Load the model
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif args.load_in_8bit or args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
else:
    device_map = None
    quantization_config = None

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=args.trust_remote_code,
    num_labels=1,
)

# Step 2: Load the dataset and pre-process it
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
train_dataset = load_dataset(args.dataset_name, split="train")


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
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


# Preprocess the dataset and filter out examples that are longer than args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
    and len(x["input_ids_rejected"]) <= args.reward_config.max_length
)

if args.eval_split == "none":
    eval_dataset = None
else:
    eval_dataset = load_dataset(args.dataset_name, split=args.eval_split)

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
        and len(x["input_ids_rejected"]) <= args.reward_config.max_length
    )


# Step 4: Define the LoraConfig
if args.use_peft:
    peft_config = args.peft_config
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args.reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.train()
