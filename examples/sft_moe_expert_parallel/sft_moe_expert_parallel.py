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
#     "transformers @ git+https://github.com/huggingface/transformers.git@ep-mesh",
#     "peft>=0.21.0",
#     "accelerate>=1.15.0",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
LoRA SFT of REDACTED on 64 H100s across 8 nodes: the experts are split 8 ways, everything else is sharded
across all 64, and each rank trains on its own slice of the batch. 14.1 s/step, 131k tokens/step, 40 GB per GPU.

    sbatch sft_moe_expert_parallel.slurm
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


# The checkpoint, staged on every node's local disk.
MODEL = "REDACTED"

# Read the dataset before loading the model: 64 GPUs holding a loaded model is the most expensive place to wait
# on I/O.
train_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train[:500000]")

training_args = SFTConfig(
    output_dir="REDACTED-SFT",
    model_init_kwargs={
        "dtype": torch.bfloat16,
        "distributed_config": DistributedConfig(
            tp_size=1,
            fsdp_size=64,
            ep_size=8,
            # The rule selects token dispatch. Its key is the experts module's full path, which on a
            # composite model carries the sub-model prefix.
            ep_plan={"model.language_model.layers.*.mlp.experts": "ep_dispatch_experts"},
        ),
    },
    per_device_train_batch_size=1,
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
