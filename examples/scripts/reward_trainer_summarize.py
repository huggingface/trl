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
from typing import Optional, Union
import torch

from datasets import load_dataset
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, PreTrainedTokenizerBase
import tyro

from trl import RewardTrainer


tqdm.pandas()

### fix from https://github.com/huggingface/trl/issues/274
# Define and parse arguments.
@dataclass
class ScriptArguments:
    model_name: str = "EleutherAI/pythia-1b"
    """the model name"""
    dataset_name: str = "CarperAI/openai_summarize_comparisons"
    """the dataset name"""
    dataset_text_field: str = "prompt"
    """the text field of the dataset"""
    train_split: str = "train"
    """the dataset split to train on"""
    eval_split: str = "test"
    """the dataset split to evaluate on; default to 'none' (no evaluation)"""
    trust_remote_code: bool = True
    """Enable `trust_remote_code`"""
    hf_trainer: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="output",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=False,
            learning_rate=1e-5,
            weight_decay=0.001,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=5,
            evaluation_strategy="epoch",
        )
    )
    seq_length: int = 560
    """the length of the post + summary"""


class GPTRewardTrainer(RewardTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_chosen = rewards[jidx]
        rewards_rejected = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss


@dataclass
class GPTRewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        # features_chosen = []
        # features_rejected = []
        merged_features = []
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch



def create_and_prepare_model(args):

    torch_dtype = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        torch_dtype=torch_dtype,
    )

    model.config.torch_dtype = torch_dtype
    model.config.use_cache = True


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def create_and_prepare_dataset(args, tokenizer, split, num_proc=2):
    dataset = load_dataset(args.dataset_name, split=split)
    original_columns = dataset.column_names

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(
                prompt + "\n\n" + chosen, padding="max_length", truncation=True, max_length=args.seq_length
            )
            tokenized_rejected = tokenizer(
                prompt + "\n\n" + rejected, padding="max_length", truncation=True, max_length=args.seq_length
            )
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    dataset = dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)

    return dataset

args = tyro.cli(ScriptArguments)
model, tokenizer = create_and_prepare_model(args)
train_dataset = create_and_prepare_dataset(args, tokenizer, args.train_split)
eval_dataset = create_and_prepare_dataset(args, tokenizer, args.eval_split)

data_collator = GPTRewardDataCollatorWithPadding(tokenizer, max_length=args.seq_length, pad_to_multiple_of=8)

trainer = GPTRewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args.hf_trainer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_length=args.seq_length,
    data_collator=data_collator,
)
trainer.train()
