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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, TrainingArguments

from trl import SFTTrainer


tqdm.pandas()


@dataclass
class ScriptArguments:
    model_name: str = "facebook/opt-350m"
    """the model name"""
    dataset_name: str = "timdettmers/openassistant-guanaco"
    """the dataset name"""
    dataset_text_field: str = "text"
    """the text field of the dataset"""
    eval_split: str = "test"
    """the dataset split to evaluate on; default to 'none' (no evaluation)"""
    load_in_8bit: bool = False
    """load the model in 8 bits precision"""
    load_in_4bit: bool = False
    """load the model in 4 bits precision"""
    trust_remote_code: bool = False
    """Enable `trust_remote_code`"""
    use_auth_token: bool = False
    """Use HF auth token to access the model"""
    seq_length: int = 512
    """the input sequence length"""
    sft_config: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="output",
            per_device_train_batch_size=64,
            gradient_accumulation_steps=16,
            learning_rate=1.41e-5,
            logging_steps=1,
            num_train_epochs=3,
            max_steps=-1,
            report_to="tensorboard",
            save_steps=100,
            save_total_limit=10,
            push_to_hub=False,
            hub_model_id=None,
        )
    )
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=64,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    generation_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            temperature=0.8,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=-1,
        )
    )


args = tyro.cli(ScriptArguments)

# Step 1: Load the model
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif args.load_in_8bit or args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
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
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Step 2: Load the dataset
dataset = load_dataset(args.dataset_name, split="train")
eval_dataset = None
if args.eval_split != "none":
    eval_dataset = load_dataset(args.dataset_name, split=args.eval_split)

# Step 3: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=args.sft_config,
    max_seq_length=args.seq_length,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    dataset_text_field=args.dataset_text_field,
    peft_config=args.peft_config if args.use_peft else None,
)
trainer.train()

# Step 4: generate predictions and log
metrics = trainer.evaluate()
trainer.log_metrics("predict", metrics)
dataloader = DataLoader(
    trainer.sample_dataset.with_format("torch").select(range(10)),
    collate_fn=trainer.data_collator,
    batch_size=1,  # only support `batch_size` for now
)
sample_output = defaultdict(list)
for data in dataloader:
    data = {key: data[key].to(trainer.args.device) for key in data}
    args.generation_config.max_new_tokens = data["input_ids"].shape[1] + data["remaining_input_ids"].shape[1]
    res = model.generate(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        generation_config=args.generation_config,
    )

    sample_output["queries"].extend(tokenizer.batch_decode(data["input_ids"]))
    sample_output["responses"].extend(tokenizer.batch_decode(res[:, data["input_ids"].shape[1] :]))
    sample_output["reference_response"].extend(tokenizer.batch_decode(data["remaining_input_ids"]))
if "wandb" in args.sft_config.report_to and trainer.accelerator.is_main_process:
    import wandb

    all_df = pd.DataFrame(sample_output)
    wandb.log({"samples/": wandb.Table(dataframe=all_df)})

# Step 4: Save the model
trainer.save_model(args.sft_config.output_dir)
