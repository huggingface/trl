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
#     "accelerate>=1.15.0",  # `fsdp_activation_checkpointing_offload`, used by the config below
# ]
# ///

"""
Fine-tune Qwen3-8B on 1,048,576-token sequences, one 8xH100 node.

Context parallelism splits each sequence across the 8 GPUs, so every GPU holds 131,072 tokens and
attention is computed as a ring. One book-length sequence per step, 373 s/step, 56.2 GB per GPU.

The accelerate config sets `fsdp_activation_checkpointing_offload`, which is what keeps an 8B model
inside 80 GB at this length. It ships in accelerate 1.15.0.

accelerate launch \
    --config_file examples/sft_qwen3_8b_1m_context/context_parallel_8gpu.yaml \
    examples/sft_qwen3_8b_1m_context/sft_qwen3_8b_1m_context.py

Swapping the model: context parallelism expresses only full causal attention, so models with
sliding-window or linear attention layers are refused. That rules out gpt-oss, Gemma 3/4, Mistral, and
Qwen3.5 and later. Qwen3 and Qwen3-MoE are full attention. Qwen3-0.6B takes 137 s/step on the same node.
"""

import accelerate
import torch
import transformers
from datasets import Dataset, load_dataset
from packaging.version import Version

from trl import SFTConfig, SFTTrainer


MODEL = "Qwen/Qwen3-8B"
SEQ_LEN = 1_048_576

# `accelerate launch` does not read the dependency header above, so the versions it declares are not
# enforced at run time. Both features below are dropped without an error when they are missing, and the
# run starts anyway, so check them here rather than let a 373 s/step job start off wrong. On transformers
# v4 the `rope_parameters` override is ignored and the model trains at 1M with no YaRN. On accelerate
# < 1.15 the config's `fsdp_activation_checkpointing_offload` is ignored and the run goes OOM.
if Version(transformers.__version__) < Version("5.0.0"):
    raise RuntimeError(
        f"This example needs transformers>=5.0 for the `rope_parameters` schema, got {transformers.__version__}."
    )
if Version(accelerate.__version__) < Version("1.15.0"):
    raise RuntimeError(
        f"This example needs accelerate>=1.15.0 for `fsdp_activation_checkpointing_offload`, got "
        f"{accelerate.__version__}."
    )


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
        # One book-length sequence is one step, so the default of 500 would log nothing at all.
        logging_steps=1,
        # `save_model` below writes the final model; an intermediate checkpoint of an 8B model with its
        # optimizer state is another 123 GB on disk.
        save_strategy="no",
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
