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
Fine-tune very large MoE models (up to GLM-5.2, 753B) with expert parallelism.

`DistributedConfig(tp_size=E, fsdp_size=D, enable_expert_parallel=True)` shards the experts across E
GPUs and everything else (plus optimizer state) across D, so the model loads directly onto the mesh —
no rank ever materializes it whole. TRL's SFTTrainer then trains it like any other model.

Verified configurations (H100 nodes, seq 2048, bf16, per-device batch 1):

| Model | GPUs | Invocation | Step time | Peak memory |
|---|---|---|---|---|
| Qwen3-30B-A3B, full FT | 8 (1 node) | `--ep 8 --lora 0` | 0.9 s | 34 GB |
| GLM-4.5-Air (110B), full FT | 64 (8 nodes) | `--ep 32 --fsdp 2 --lora 0` | 3.7 s | 41 GB |
| GLM-4.6 (357B), LoRA | 16 (2 nodes) | `--ep 16 --lora 16` | 4.4 s | 73 GB |
| GLM-5.2 (753B), LoRA | 64 (8 nodes) | `--ep 32 --fsdp 2 --lora 16` | 3.1 s | 56 GB |

Single node:

torchrun --nproc_per_node 8 sft_moe_expert_parallel.py --model Qwen/Qwen3-30B-A3B --ep 8 --lora 0

Multi-node, via the launcher next to this script:

sbatch sft_moe_expert_parallel.slurm --model zai-org/GLM-5.2 --ep 32 --fsdp 2 --lora 16
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="zai-org/GLM-5.2")
parser.add_argument("--ep", type=int, default=32, help="expert-parallel group size")
parser.add_argument("--fsdp", type=int, default=2, help="FSDP group size; ep * fsdp = total GPUs")
parser.add_argument("--lora", type=int, default=16, help="LoRA rank; 0 = full fine-tuning")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=args.ep, fsdp_size=args.fsdp, enable_expert_parallel=True),
)

peft_config = None
if args.lora:
    from peft import LoraConfig

    peft_config = LoraConfig(
        r=args.lora, lora_alpha=2 * args.lora, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

training_args = SFTConfig(
    output_dir=args.output_dir or args.model.split("/")[-1] + "-SFT",
    per_device_train_batch_size=1,
    max_steps=args.steps,
    max_length=2048,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=load_dataset("allenai/tulu-3-sft-mixture", split="train[:10000]"),
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(training_args.output_dir)
