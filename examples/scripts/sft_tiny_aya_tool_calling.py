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
#     "liger-kernel",
#     "trackio",
# ]
# ///

"""
Teach tool calling to CohereLabs/tiny-aya-global using SFT with QLoRA on the bebechien/SimpleToolCalling dataset.

The model used in this script does not have native tool-calling support. We extend its existing Jinja2 chat template to
serialize tool schemas into the system preamble and render tool calls as structured <tool_call> XML inside the model's
native <|START_RESPONSE|> / <|END_RESPONSE|> delimiters. The modified template is saved with the tokenizer, so
inference only requires loading the tokenizer from the output directory and calling apply_chat_template with
tools=TOOLS — no manual system-prompt construction needed.

Example:

    python examples/scripts/sft_tiny_aya_tool_calling.py
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from trl import SFTConfig, SFTTrainer


# These are the tool schemas that are used in the dataset
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search internal company documents, policies and project data.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "query string"}},
                "required": ["query"],
            },
            "return": {"type": "string"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Search public information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "query string"}},
                "required": ["query"],
            },
            "return": {"type": "string"},
        },
    },
]


def create_conversation(sample):
    return {
        "prompt": [{"role": "user", "content": sample["user_content"]}],
        "completion": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": sample["tool_name"],
                            "arguments": json.loads(sample["tool_arguments"]),
                        },
                    }
                ],
            },
        ],
        "tools": TOOLS,
    }


def main():
    model_id = "CohereLabs/tiny-aya-global"
    dataset_name = "bebechien/SimpleToolCalling"
    output_dir = "tiny-aya-global-tool-calling-SFT"

    # Load and format dataset
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(create_conversation, remove_columns=dataset.features)
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
        # Use the tool-aware chat template
        chat_template_path=str(Path(__file__).parent / "tiny_aya_chat_template.jinja"),
        warmup_steps=5,
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
        peft_config=peft_config,
    )
    trainer.train()

    # Save model and tokenizer (tokenizer carries the updated chat template)
    trainer.save_model(output_dir)
    trainer.push_to_hub(dataset_name=dataset_name)


if __name__ == "__main__":
    main()
