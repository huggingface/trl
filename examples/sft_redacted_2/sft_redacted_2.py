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
#     "trl>=1.16.0",
#     "transformers @ git+https://github.com/huggingface/transformers.git@REDACTED",
#     "peft>=0.21.0",
#     "accelerate>=1.15.0",
#     "flash-attn-4",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
LoRA SFT of REDACTED on one 8xB300 node: the experts are split 8 ways, everything else is sharded across the 8 GPUs,
and each rank trains on its own slice of the batch. One epoch is 36 steps of 1.75 s, 32.8k tokens per step,
223 GiB per GPU.

    sbatch sft_redacted_2.slurm
"""

from datasets import load_dataset
from peft import LoraConfig
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


# The checkpoint, staged on the node's local disk.
MODEL = "REDACTED"

# Assistant turns carry their reasoning in a `thinking` field, which the chat template renders into the analysis
# channel.
train_dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

training_args = SFTConfig(
    output_dir="REDACTED-SFT",
    model_init_kwargs={
        # TRL loads a model id in float32 unless told otherwise; "auto" keeps the checkpoint's dtype.
        "dtype": "auto",
        # Flash Attention 4 runs on Blackwell and keeps packed sequences apart.
        "attn_implementation": "flash_attention_4",
        # The model's own expert plan selects token dispatch, so the sizes are the whole configuration.
        "distributed_config": DistributedConfig(fsdp_size=8, ep_size=8),
    },
    per_device_train_batch_size=1,
    # B300s have the memory to keep the activations: 1.4x faster than recomputing them.
    gradient_checkpointing=False,
    num_train_epochs=1,
    max_length=4096,
    # bfd packing flattens the batch (padding-free) and passes position_ids, which the attention uses to keep the
    # packed sequences apart.
    packing=True,
    # Train on the assistant turns only (reasoning, answer and the end-of-turn token).
    assistant_only_loss=True,
    learning_rate=1e-4,
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
