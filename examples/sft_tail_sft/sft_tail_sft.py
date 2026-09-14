# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

# /// script
# dependencies = [
#     "trl",
# ]
# ///

"""Minimal TailSFT example based on https://huggingface.co/papers/2608.25756."""

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = Dataset.from_dict(
    {
        "prompt": [
            "Complete the sequence: 1, 1, 2, 3, 5,",
            "What is the capital of France?",
            "Write a Python expression that adds one to x.",
            "What is 12 * 12?",
        ],
        "completion": [" 8", " Paris", " x + 1", " 144"],
    }
)

# Prepare the data once so initial-policy scoring and training use exactly the same tokenization and loss mask.
preparation_args = SFTConfig(
    output_dir="tail-sft-preparation",
    completion_only_loss=True,
    loss_type="nll",
    max_length=128,
    report_to="none",
)
preparation_trainer = SFTTrainer(
    model=model,
    args=preparation_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
dataset = preparation_trainer.train_dataset

# Algorithm 1 records the initial policy's mean target-token cross-entropy for every sequence.
initial_losses = []
model.eval()
for batch in DataLoader(dataset, batch_size=2, collate_fn=preparation_trainer.data_collator):
    batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[..., :-1, :]
        labels = batch["labels"][..., 1:]
        loss_mask = labels != -100
        per_token_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
        initial_losses.extend(((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1)).tolist())
dataset = dataset.add_column("initial_loss", initial_losses)

training_args = SFTConfig(
    output_dir="tail-sft-example",
    loss_type="tail_sft",
    tail_sft_filter_fraction=0.5,
    tail_sft_filter_schedule="ramp",
    max_steps=4,
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    logging_steps=1,
    report_to="none",
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
