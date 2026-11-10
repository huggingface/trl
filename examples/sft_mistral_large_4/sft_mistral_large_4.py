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
#     "trl>=1.17.0",
#     "transformers>=5.22.0",
#     "peft>=0.21.0",
#     "accelerate>=1.15.0",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
LoRA SFT of Mistral Large 4 on one 8xB300 node, on top of its FP8 weights: the experts are split 8 ways, everything
else is sharded across the 8 GPUs, and each rank trains on its own slice of the batch. 32.7k tokens per step of ~8 s,
142 GiB per GPU.

    sbatch sft_mistral_large_4.slurm
"""

from datasets import load_dataset
from peft import LoraConfig
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


MODEL = "mistralai/Mistral-Large-4-1T-A52B"

# Assistant turns carry their reasoning in a `thinking` field, so the conversations are rendered with high reasoning
# effort, the mode in which the model reasons.
train_dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
train_dataset = train_dataset.map(lambda x: {"chat_template_kwargs": {"reasoning_effort": "high"}})

training_args = SFTConfig(
    output_dir="Mistral-Large-4-SFT",
    model_init_kwargs={
        # TRL loads a model id in float32 unless told otherwise; "auto" keeps the checkpoint's dtype, and LoRA trains
        # on top of the FP8 weights.
        "dtype": "auto",
        # The model's own expert plan selects token dispatch, so the sizes are the whole configuration.
        "distributed_config": DistributedConfig(fsdp_size=8, ep_size=8),
    },
    per_device_train_batch_size=1,
    num_train_epochs=1,
    max_length=4096,
    # bfd packing flattens the batch (padding-free) and passes position_ids, which the attention uses to keep the
    # packed sequences apart.
    packing=True,
    # Train on the assistant turns only (reasoning, answer and the end-of-turn token).
    assistant_only_loss=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=1,
    # Resuming is not supported for models sharded at load time, so the optimizer state cannot be checkpointed.
    save_only_model=True,
    report_to="trackio",
)
trainer = SFTTrainer(
    model=MODEL,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        # Attention and the dense and shared-expert MLPs of the language model; the routed experts stay frozen.
        target_modules=r".*language_model.*\.(q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|gate_proj|up_proj|down_proj)",
    ),
)
trainer.train()
