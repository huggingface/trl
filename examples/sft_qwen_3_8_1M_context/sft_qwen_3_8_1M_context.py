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
#     "trackio",
#     "transformers>=5.0",  # the `rope_parameters` override below is a v5 schema; v4 ignores it silently
# ]
# ///

"""
Fine-tune Qwen3-8B on 1,048,576-token sequences, one 8xH100 node.

Context parallelism splits each sequence across the 8 GPUs, so every GPU holds 131,072 tokens and
attention is computed as a ring. One book-length sequence per step, 364 s/step, 56.2 GB per GPU.

accelerate launch \
    --config_file examples/sft_qwen_3_8_1M_context/context_parallel_8gpu.yaml \
    examples/sft_qwen_3_8_1M_context/sft_qwen_3_8_1M_context.py

Swapping the model: context parallelism expresses only full causal attention, so models with
sliding-window or linear attention layers are refused. That rules out gpt-oss, Gemma 3/4, Mistral, and
Qwen3.5 and later. Qwen3 and Qwen3-MoE are full attention. Measured on the same node: Qwen3-0.6B 137 s,
Qwen3-30B-A3B 483 s, Qwen3-32B 1295 s (the last one also needs the MLP tiled over the sequence).
"""

import torch
from datasets import Dataset, load_dataset

from trl import SFTConfig, SFTTrainer


MODEL = "Qwen/Qwen3-8B"
SEQ_LEN = 1_048_576


def build_dataset(rows=4, chars_per_row=6_000_000):
    """Join PG-19 books into rows long enough to fill a million-token sequence.

    Books are far shorter than that, so they are concatenated and the trainer truncates to
    `max_length`. This is ordinary continued-pretraining practice, and it is *not* `packing=True`:
    packing promises documents cannot attend to each other, which context parallelism cannot honour
    and therefore refuses.
    """
    stream = load_dataset("emozilla/pg19", split="train", streaming=True)

    texts, current = [], ""
    for book in stream:
        current += book["text"]
        if len(current) >= chars_per_row:
            texts.append(current)
            current = ""
            if len(texts) == rows:
                break
    return Dataset.from_dict({"text": texts})


def main():
    training_args = SFTConfig(
        output_dir="Qwen3-8B-1M",
        max_length=SEQ_LEN,
        per_device_train_batch_size=1,
        # The sequence has to divide `cp_size * 2`, which the collator handles by padding.
        pad_to_multiple_of=16,
        # Activation checkpointing is already enabled in the accelerate config; setting both raises.
        gradient_checkpointing=False,
        bf16=True,
        # A base model used far beyond its trained context needs RoPE scaling: without it, loss at 1M
        # starts around 10.6 instead of 4.4. In Transformers v5 `rope_theta` lives inside
        # `rope_parameters`, so the override has to carry it.
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1_000_000,
                "factor": 32.0,
                "original_max_position_embeddings": 32768,
            },
        },
    )

    trainer = SFTTrainer(model=MODEL, args=training_args, train_dataset=build_dataset())
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
