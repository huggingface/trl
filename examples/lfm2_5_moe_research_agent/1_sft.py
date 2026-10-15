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
#     "trl>=1.14.0",
#     "transformers>=5.19.0",  # the LFM2-MoE expert plan defaults to token dispatch
#     "datasets>=5.0.0",  # `Json` feature
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
Full SFT of LiquidAI/LFM2.5-8B-A1B on function calling, on one 8xH100 node: the 32 experts are split 8 ways (expert
parallelism with token dispatch), everything else is FSDP-sharded across the 8 GPUs, and the loss is on the assistant
tool calls only.

    torchrun --nproc_per_node 8 examples/lfm2_5_moe_research_agent/1_sft.py

LiquidAI/LFM2.5-24B-A2B has the same architecture: set `MODEL` to it to train it with the same layout.
"""

import json

from datasets import Features, Json, load_dataset
from transformers.distributed import DistributedConfig

from trl import SFTConfig, SFTTrainer


MODEL = "LiquidAI/LFM2.5-8B-A1B"


def to_conversation(example):
    # xLAM lists each tool's parameters as `{name: {type, description, default}}`: wrap them in the function schema the
    # chat template renders into the system prompt
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {"type": "object", "properties": tool["parameters"]},
            },
        }
        for tool in json.loads(example["tools"])
    ]
    tool_calls = [{"type": "function", "function": call} for call in json.loads(example["answers"])]
    return {
        "messages": [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ],
        "tools": tools,
    }


# Read the dataset before loading the model: 8 GPUs holding a loaded model is the most expensive place to wait on I/O.
dataset = load_dataset("argilla/apigen-function-calling", split="train")
# The 60k xLAM samples (https://huggingface.co/papers/2406.18518): queries answered by one or several parallel calls
dataset = dataset.filter(lambda example: example["origin"] == "xLAM")
# Every row calls different tools with different arguments, which Arrow can only store as JSON
dataset = dataset.map(
    to_conversation,
    remove_columns=dataset.column_names,
    features=Features({"messages": Json(), "tools": Json()}),
)

training_args = SFTConfig(
    output_dir="LFM2.5-8B-A1B-SFT-function-calling",
    model_init_kwargs={
        # FSDP keeps float32 master weights and casts them to bfloat16 for compute, so the optimizer steps are not
        # rounded away as they would be on bfloat16 weights.
        "dtype": "float32",
        # The model's own expert plan selects token dispatch, so the sizes are the whole configuration.
        "distributed_config": DistributedConfig(fsdp_size=8, ep_size=8, fsdp_mixed_precision=True),
    },
    assistant_only_loss=True,
    # Padded batches, not packing: without the `causal_conv1d` kernel, the short convolutions of the LFM2 layers
    # ignore `seq_idx` and would mix the last tokens of one packed sequence into the first tokens of the next.
    per_device_train_batch_size=8,
    max_length=2048,
    learning_rate=1e-5,
    num_train_epochs=1,
    dataset_num_proc=16,
    logging_steps=1,
    # Resuming is not supported for models sharded at load time, so the optimizer state cannot be checkpointed.
    save_only_model=True,
    report_to="trackio",
)
trainer = SFTTrainer(model=MODEL, args=training_args, train_dataset=dataset)
trainer.train()
