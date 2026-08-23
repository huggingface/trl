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
LoRA fine-tune GLM-5.2 (753B) on 64 H100s across 8 nodes.

`DistributedConfig(tp_size=32, fsdp_size=2, enable_expert_parallel=True)` shards the experts across
32 GPUs and everything else across 2, so the model loads directly onto the mesh — no rank ever
materializes it whole. Measured: 3.1 s/step at sequence length 2048, 56 GB peak per GPU, loss
3.3 -> 2.3 in 50 steps on tulu-3 chat data; the adapter saves in seconds.

Launch from this directory:

sbatch sft_glm_5_2_expert_parallel.slurm
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-5.2",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=32, fsdp_size=2, enable_expert_parallel=True),
)

training_args = SFTConfig(
    output_dir="GLM-5.2-SFT-LoRA",
    per_device_train_batch_size=1,
    max_steps=50,
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
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
)
trainer.train()
trainer.save_model(training_args.output_dir)
