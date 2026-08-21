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
# ]
# ///

"""
Fine-tune on 1M-token sequences with context parallelism.

One 8xH100 node, one book-length sequence per step. The sequence is split across the 8 GPUs, so each
one holds 131,072 tokens and attention is computed as a ring.

Example:

accelerate launch \
    --config_file examples/sft_qwen_3_8_1M_context/context_parallel_8gpu.yaml \
    examples/sft_qwen_3_8_1M_context/sft_qwen_3_8_1M_context.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --dataset_name emozilla/pg19 \
    --max_length 1048576 \
    --per_device_train_batch_size 1 \
    --pad_to_multiple_of 16 \
    --gradient_checkpointing False \
    --bf16 \
    --output_dir Qwen3-8B-1M

Measured on 8xH100 (step time, peak memory per GPU):

    Qwen3-0.6B        137 s   27.9 GB
    Qwen3-8B          364 s   56.2 GB
    Qwen3-30B-A3B     483 s   46.0 GB
    Qwen3-32B        1295 s   60.8 GB

Two constraints worth knowing before swapping the model:

- Context parallelism only expresses full causal attention, so models with sliding-window or linear
  attention layers are refused. That rules out gpt-oss, Gemma 3/4, Mistral, and Qwen3.5 and later
  (three quarters of whose layers are linear attention). Qwen3 and Qwen3-MoE are full attention.
- Packing is refused for the same reason: it relies on a block-diagonal mask.
"""

import torch
from datasets import load_dataset

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser


def main(script_args, training_args, model_args):
    # A base model used far beyond its trained context needs RoPE scaling: without it, loss at 1M starts
    # around 10.6 instead of 4.4. In Transformers v5 `rope_theta` lives inside `rope_parameters`, so the
    # override has to carry it.
    training_args.model_init_kwargs = {
        "dtype": torch.bfloat16,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 1_000_000,
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
        },
    }

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
