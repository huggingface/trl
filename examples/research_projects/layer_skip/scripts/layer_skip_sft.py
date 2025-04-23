# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import config
import torch
from custom_trainer import LayerSkipSFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import DataCollatorForCompletionOnlyLM, SFTConfig


def formatting_prompts_func(example):
    text = f"### Instruction: {example['utterance']}\n ### Response: {example['semantic_parse']}"

    # Inject eos_token as a string before tokenization, because they are not always added
    # See: https://github.com/huggingface/transformers/issues/22794 and
    # https://github.com/huggingface/trl/issues/1623
    if tokenizer.eos_token:  # usually something like "</s>" for GPT2 or "<|endoftext|>"
        text += f"{tokenizer.eos_token}"

    return text


if __name__ == "__main__":
    # load the dataset
    print("[INFO] loading the dataset...")
    train_dataset = load_dataset(config.dataset_name, split="train")

    print(f"output_root_dir: {config.output_root_dir}")
    print(f"hub_model_id: {config.hub_model_id}")

    # load the model and tokenizer
    print("[INFO] loading the model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, add_eos_token=True)

    # adding pad and eos tokens if not provided in the tokenizer
    if tokenizer.pad_token is None:
        # Add '[PAD]' token if it doesn't exist
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.eos_token is None or tokenizer.eos_token == tokenizer.bos_token:
        # Add '[EOS]' token if it doesn't exist
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.eos_token_id = tokenizer.eos_token_id

    response_template = " ### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    args = SFTConfig(
        do_train=True,
        bf16=True,
        max_seq_length=None,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        packing=False,
        num_train_epochs=1.0,
        report_to="none",
        push_to_hub=True,
        hub_model_id=config.hub_model_id,
        output_dir=config.output_dir,
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
    )

    trainer = LayerSkipSFTTrainer(
        model,
        train_dataset=train_dataset,
        args=args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()
