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
#     "trl[peft]",
#     "bitsandbytes",
#     "trackio",
# ]
# ///

"""
Teach tool calling to CohereLabs/tiny-aya-global using SFT with QLoRA on the
bebechien/SimpleToolCalling dataset.

The model used in this script does not have native tool-calling support.
We embed the tool schemas directly in the system prompt and train the model
to produce a structured JSON response. This technique can work with any base
language model regardless of its chat template.

Example:

    python examples/scripts/sft_tiny_aya_tool_calling.py
"""

import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import get_json_schema

from trl import SFTConfig, SFTTrainer


# --- Tool definitions ---
def search_knowledge_base(query: str) -> str:
    """
    Search internal company documents, policies and project data.

    Args:
        query: query string
    """
    return "Internal Result"


def search_google(query: str) -> str:
    """
    Search public information.

    Args:
        query: query string
    """
    return "Public Result"


TOOLS = [get_json_schema(search_knowledge_base), get_json_schema(search_google)]
TOOLS_TEXT = json.dumps([t["function"] for t in TOOLS], indent=2)

SYSTEM_MSG = (
    "You are a helpful assistant with access to tools. "
    "For every user request, you MUST respond ONLY with a JSON object selecting the right tool. "
    "Never answer from your own knowledge.\n\n"
    f"Available tools:\n{TOOLS_TEXT}\n\n"
    'Respond exclusively in this JSON format: {"name": "<tool_name>", "arguments": {"<arg>": "<value>"}}'
)


def create_conversation(sample):
    tool_call_content = json.dumps(
        {
            "name": sample["tool_name"],
            "arguments": json.loads(sample["tool_arguments"]),
        }
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": sample["user_content"]},
            {"role": "assistant", "content": tool_call_content},
        ]
    }


def main():
    model_id = "CohereLabs/tiny-aya-global"
    dataset_name = "bebechien/SimpleToolCalling"
    output_dir = "tiny-aya-global-tool-calling-SFT"

    # Load and format dataset
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    dataset = dataset.train_test_split(test_size=0.5, shuffle=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Train
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=1,
        report_to="trackio",
        trackio_space_id=output_dir,
        max_length=1024,
        use_liger_kernel=True,
        activation_offloading=True,
        push_to_hub=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(output_dir)
    trainer.push_to_hub(dataset_name=dataset_name)


if __name__ == "__main__":
    main()
