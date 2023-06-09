# 0. imports
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from datasets import load_dataset

from trl import DPOTrainer


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
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

# 2. initialize trainer
dpo_config = {
    "batch_size": 1,
}
# TODO
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    beta=0.1,
)

# 3. encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = dpo_trainer.generate(
    [item for item in query_tensor], return_prompt=False, **generation_kwargs
)
response_txt = tokenizer.decode(response_tensor[0])

# 5. train model with dpo which uses the reward implicitly defined by the model and model_ref
train_stats = dpo_trainer.step([query_tensor[0]], [response_tensor[0]])
