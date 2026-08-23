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

"""
Full fine-tune GLM-4.5-Air (110B) — every parameter, no LoRA — on 64 H100s across 8 nodes.

`DistributedConfig(tp_size=32, fsdp_size=2, enable_expert_parallel=True)` shards the experts across
32 GPUs and everything else across 2; parameters, gradients, and optimizer state all divide by the
mesh, which is what makes full fine-tuning of a 110B model peak at 41 GB per GPU. Measured:
3.7 s/step at sequence length 2048, loss 3.35 -> 2.13 in 20 steps on tulu-3 chat data; the final
`save_model` gathers and writes the full 206 GiB checkpoint in standard HF format (~19 minutes).

Launch from this directory:

sbatch sft_glm_4_5_air_full_finetune.slurm
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-4.5-Air",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=32, fsdp_size=2, enable_expert_parallel=True),
)

training_args = SFTConfig(
    output_dir="GLM-4.5-Air-SFT",
    per_device_train_batch_size=1,
    max_steps=20,
    max_length=2048,
    logging_steps=1,
    # Mid-training checkpoints are for resuming, and resume is not yet supported for models
    # sharded at load time; the final weights are saved explicitly below.
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=load_dataset("allenai/tulu-3-sft-mixture", split="train[:10000]"),
)
trainer.train()
trainer.save_model(training_args.output_dir)
