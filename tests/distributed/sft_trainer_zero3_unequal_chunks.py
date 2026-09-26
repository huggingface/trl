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

import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, default_data_collator

from trl import SFTConfig, SFTTrainer


def main():
    torch.manual_seed(2026)
    rank = int(os.environ["RANK"])
    sequence_length = 384
    input_ids = [3 + index % 29 for index in range(sequence_length)]
    labels = [-100] * sequence_length
    width = 16 if rank == 0 else 80
    for island in range(4):
        start = 1 + island * 96
        labels[start : start + width] = input_ids[start : start + width]
    dataset = Dataset.from_dict(
        {
            "input_ids": [input_ids] * 4,
            "attention_mask": [[1] * sequence_length] * 4,
            "labels": [labels] * 4,
        }
    )

    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    processing_class = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>")
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=1024,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            tie_word_embeddings=False,
            attn_implementation="eager",
            pad_token_id=0,
            eos_token_id=1,
        )
    ).to(torch.bfloat16)
    args = SFTConfig(
        output_dir=sys.argv[1],
        max_steps=1,
        seed=2026,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        loss_type="chunked_nll",
        deepspeed={
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "train_batch_size": "auto",
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 0,
                "stage3_model_persistence_threshold": 0,
            },
        },
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        bf16=True,
        report_to="none",
        save_strategy="no",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=processing_class,
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        data_collator=default_data_collator,
    )
    batch = next(iter(trainer.get_train_dataloader()))
    assert (batch["labels"] != -100).sum().item() == (64 if rank == 0 else 320)
    trainer.train()
    assert trainer.state.global_step == 1


if __name__ == "__main__":
    main()
