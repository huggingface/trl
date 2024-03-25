# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Run the KTO training script with the commands below. In general, the optimal configuration for KTO will be similar to that of DPO.

# Full training:
python examples/scripts/kto.py \
    --model_name_or_path=stabilityai/stablelm-2-zephyr-1_6b \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="kto-aligned-model" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step

# LoRA:
python examples/scripts/kto.py \
    --model_name_or_path=stabilityai/stablelm-2-zephyr-1_6b \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="kto-aligned-model-lora" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    dataset_name: str = "trl-lib/kto-mix-14k"


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer must have a chat template in order to format the examples. Alternatively, adjust this script to format the examples differently."
        )

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)

    # Apply chat template
    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
        return example

    formatted_dataset = dataset.map(format_dataset)

    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
    kto_trainer.push_to_hub()
