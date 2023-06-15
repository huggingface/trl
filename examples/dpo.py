# 0. imports
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    TrainingArguments,
    HfArgumentParser
)
from transformers.utils import PaddingStrategy
from datasets import load_dataset

from trl import DPOTrainer

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with DPO
    """
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the PPO minibatch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"})
    max_length: Optional[int] = field(
        default = 512, metadata={"help": "max length of each samples"}
    )
    label_pad_token_id: Optional[int] = field(
        default = -100, metadata={"help": "label for non response tokens"}
    )
    
    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


SANITY = True

# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples, tokenizer, max_length, label_pad_token_id):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "labels_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "labels_rejected": []
    }
    
    for question, response_j, response_k in zip(
        examples["question"], examples["response_j"], examples["response_k"]
    ):
    
        input_ids_query = tokenizer(question)["input_ids"]
        input_ids_response_chosen = tokenizer(response_j)["input_ids"]
        input_ids_response_rejected = tokenizer(response_k)["input_ids"]
        len_query = len(input_ids_query)

        input_ids_chosen = input_ids_query + input_ids_response_chosen
        attention_mask_chosen = [1] * len(input_ids_chosen)
        labels_chosen = [label_pad_token_id]*len_query + input_ids_chosen[len_query:]
        input_ids_rejected = input_ids_query + input_ids_response_rejected
        attention_mask_rejected = [1] * len(input_ids_rejected)
        labels_rejected = [label_pad_token_id]*len_query + input_ids_rejected[len_query:]
        
        # truncate to max_length 
        new_examples["input_ids_chosen"].append(input_ids_chosen[-max_length:])
        new_examples["attention_mask_chosen"].append(attention_mask_chosen[-max_length:])
        new_examples["labels_chosen"].append(labels_chosen[-max_length:])
        new_examples["input_ids_rejected"].append(input_ids_rejected[-max_length:])
        new_examples["attention_mask_rejected"].append(attention_mask_rejected[-max_length:])
        new_examples["labels_rejected"].append(labels_rejected[-max_length:])
        
    return new_examples


# 1. load a pretrained model
model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
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
    num_proc=os.cpu_count(),
    fn_kwargs=dict(
                tokenizer=tokenizer,
                max_length=script_args.max_length,
                label_pad_token_id=script_args.label_pad_token_id
        )
)

# 2. initialize training arguments:
training_args = TrainingArguments(
    per_device_train_batch_size=script_args.batch_size,
    max_steps=10,
    remove_unused_columns=False,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    evaluation_strategy="steps",
    output_dir="./test",
    report_to = script_args.log_with
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
