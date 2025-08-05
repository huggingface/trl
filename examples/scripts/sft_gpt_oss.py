# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
Requires torch>=2.8.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/sft_gpt_oss.py 

# TODO: test with FSDP
accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml examples/scripts/sft_gpt_oss.py 
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Mxfp4Config

from trl import SFTConfig, SFTTrainer


# Load the dataset
dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

# Define the training arguments
training_args = SFTConfig(
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=1,
    per_device_train_batch_size=4,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="gpt-oss-20b-multilingual-reasoner",
    report_to="trackio",
    push_to_hub=False,  # TODO: set to True
)

# Load the model with quantization
model_name = "/fsx/vb/new-oai/gpt-oss-20b-trfs-latest"  # TODO: chat to "openai/gpt-oss-20b"
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=True),
    use_cache=False,
)
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

# Instantiate the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train
trainer.train()

trainer.save_model(training_args.output_dir)
# if training_args.push_to_hub: # TODO uncomment to push to hub
#     trainer.push_to_hub(dataset_name="HuggingFaceH4/Multilingual-Thinking")
