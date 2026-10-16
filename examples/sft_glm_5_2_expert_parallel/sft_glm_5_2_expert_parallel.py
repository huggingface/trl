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
#     "trl>=1.14.0",  # adds the MoE aux loss only when the config declares a coefficient
#     "transformers @ git+https://github.com/huggingface/transformers.git@main",
#     "peft>=0.21.0",
# ]
# ///

"""
LoRA fine-tune GLM-5.2 (753B) on 64 H100s across 8 nodes: the experts are split 8 ways with token dispatch,
everything else is sharded across all 64, and each rank trains on its own slice of the batch. The model loads
directly onto the mesh, no rank ever materializes it whole, and the adapter saves in seconds.

For the throughput levers on top of this config (packing, gradient accumulation, batch sizing), see "Making it
fast" in docs/source/distributing_training.md.

Launch from this directory:

    sbatch sft_glm_5_2_expert_parallel.slurm
"""

import torch
import transformers
from datasets import load_dataset
from packaging.version import Version
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


# `ep_size` is not in a released transformers yet. Checked before the dataset is read, so 64 ranks fail in a second
# rather than after preprocessing. Temporary: once it ships, pin `transformers>=5.19.0` in the header above and drop
# this.
if Version(transformers.__version__) < Version("5.19.0.dev0"):
    raise RuntimeError(
        f"This example needs expert parallelism, which is not in a released transformers yet. Install transformers "
        f"from main (expert parallelism merged in #48873). Got {transformers.__version__}."
    )


model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-5.2",
    dtype=torch.bfloat16,
    # The model's own expert plan selects token dispatch, so the sizes are the whole configuration.
    distributed_config=DistributedConfig(fsdp_size=64, ep_size=8),
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
