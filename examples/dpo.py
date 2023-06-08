# 0. imports
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from datasets import load_dataset

from trl import DPOTrainer


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class DPODataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_j["input_ids"],
            "attention_mask_chosen": batch_j["attention_mask"],
            "input_ids_rejected": batch_k["input_ids"],
            "attention_mask_rejected": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
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

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

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
