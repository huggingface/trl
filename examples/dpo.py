# 0. imports
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    GPT2Tokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from datasets import load_dataset

from trl import DPOTrainer


SANITY = True


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for question, response_j, response_k in zip(
        examples["question"], examples["response_j"], examples["response_k"]
    ):
        tokenized_j = tokenizer(
            "Question: " + question + "\n\nAnswer: " + response_j, truncation=True
        )
        tokenized_k = tokenizer(
            "Question: " + question + "\n\nAnswer: " + response_k, truncation=True
        )

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples


# 1. load a pretrained model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model_ref = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Load the human stack-exchange paired dataset
train_dataset = load_dataset(
    "lvwerra/stack-exchange-paired", data_dir="data/reward", split="train"
)

if SANITY:
    train_dataset = train_dataset.select(range(1000))

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    num_proc=os.cpu_count()
)

# 2. initialize training arguments:
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    max_steps=3,
    remove_unused_columns=False,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    evaluation_strategy="steps",
    output_dir="./test"
)


# 3. initialize the DPO trainer
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# 4. train
dpo_trainer.train()
