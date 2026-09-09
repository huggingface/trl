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
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "transformers @ git+https://github.com/huggingface/transformers.git@ep-fsdp-2d-mesh",
#     "peft @ git+https://github.com/huggingface/peft.git",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
LoRA SFT of a large MoE on 64 H100s across 8 nodes.

`DistributedConfig(tp_size=8, fsdp_size=8, enable_expert_parallel=True, expert_parallel_dispatch=True)` splits the
experts across 8 GPUs and shards everything across all 64 with FSDP2 on a 2-D mesh; with dispatch every rank
trains on its own slice of the batch and tokens travel to the rank owning their expert. Measured: 22.3 s/step at
per-device batch 4 (23k tokens/s), 43.7 s with gradient accumulation 2, loss 1.26 -> 0.6 over 100 steps on tulu-3.

Stage the checkpoint on node-local disk first: sharded loading took 29 s from NVMe and did not finish in 40 min
from a network mount. Gradient checkpointing has to stay on: weights, optimizer state and FSDP buffers already
take 66-70 GB per GPU. Needs transformers `ep-fsdp-2d-mesh` (#48204) and peft main (#3578), as the header pins.

Launch from this directory:

    sbatch sft_moe_expert_parallel.slurm
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


# The checkpoint, staged on every node's local disk. The model this was measured on is not public.
MODEL = "REDACTED"

# Read the dataset before loading the model: 64 GPUs holding a loaded model is the most expensive place to wait
# on I/O.
train_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train[:500000]")

training_args = SFTConfig(
    output_dir="sft_moe_expert_parallel",
    model_init_kwargs={
        "dtype": torch.bfloat16,
        "distributed_config": DistributedConfig(
            tp_size=8, fsdp_size=8, enable_expert_parallel=True, expert_parallel_dispatch=True
        ),
    },
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_steps=500,
    max_length=2048,
    # bfd packing flattens the batch (padding-free) and passes position_ids; transformers builds the
    # block-diagonal mask from them, so sequences do not attend to each other under sdpa either.
    packing=True,
    logging_steps=1,
    # Resuming is not supported for models sharded at load time, so the optimizer state cannot be checkpointed.
    save_only_model=True,
    report_to="trackio",
)
trainer = SFTTrainer(
    model=MODEL,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
)
trainer.train()
