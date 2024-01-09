# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
python examples/scripts/sft.py \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="output" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --save_steps=100 \
    --save_total_limit=10 \
    --push_to_hub \
    --gradient_checkpointing \
    --model_name="facebook/opt-350m" \
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer, is_xpu_available


tqdm.pandas()


@dataclass
class ScriptArguments:
    model_name: str = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_text_field: str = field(default="text", metadata={"help": "the text field of the dataset"})
    seq_length: int = field(default=512, metadata={"help": "Input sequence length"})
    load_in_8bit: bool = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    use_auth_token: bool = field(default=True, metadata={"help": "Use HF auth token to access the model"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    peft_lora_r: int = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: int = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "Target modules for LoRA adapters"})


parser = HfArgumentParser((ScriptArguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()
training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)


# Step 1: Load the model
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif args.load_in_8bit or args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=args.use_auth_token,
)

# Step 2: Load the dataset
dataset = load_dataset(args.dataset_name, split="train")

# # Step 3: Define the training arguments
# training_args = TrainingArguments(
#     output_dir=script_args.output_dir,
#     per_device_train_batch_size=script_args.batch_size,
#     gradient_accumulation_steps=script_args.gradient_accumulation_steps,
#     learning_rate=script_args.learning_rate,
#     logging_steps=script_args.logging_steps,
#     num_train_epochs=script_args.num_train_epochs,
#     max_steps=script_args.max_steps,
#     report_to=script_args.report_to,
#     save_steps=script_args.save_steps,
#     save_total_limit=script_args.save_total_limit,
#     push_to_hub=script_args.push_to_hub,
#     hub_model_id=script_args.hub_model_id,
#     gradient_checkpointing=script_args.gradient_checkpointing,
#     # TODO: uncomment that on the next release
#     # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
# )

# Step 4: Define the LoraConfig
if args.use_peft:
    peft_config = LoraConfig(
        r=args.peft_lora_r,
        lora_alpha=args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
else:
    peft_config = None

# Step 5: Define the Trainer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=args.seq_length,
    train_dataset=dataset,
    dataset_text_field=args.dataset_text_field,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(args.output_dir)
