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

import ast
from dataclasses import dataclass, field

from datasets import load_dataset
from huggingface_hub import ModelCard
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/llava-instruct-mix"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int`, *optional*):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/llava-instruct-mix",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def process_example(example):
    messages = []
    for message in ast.literal_eval(example["conversations"]):
        content = message["value"]
        content = content.replace("<image>", "").strip()
        role = "user" if message["from"] == "human" else "assistant"
        messages.append({"role": role, "content": content})
    return {"messages": messages, "images": [example["image"]]}


def filter_long_examples(example):
    total_length = sum(len(msg["content"]) for msg in example["messages"])
    return total_length <= 1000


def split_prompt_completion(example):
    """
    Splits the messages into a prompt and a completion. The last message is considered the completion.
    """
    assert len(example["messages"]) > 1
    example["prompt"] = example["messages"][:-1]
    example["completion"] = example["messages"][-1:]
    return example


model_card = ModelCard("""
---
tags: [trl]
---

# LLaVA Instruct Mix

## Summary

The LLaVA Instruct Mix dataset is a processed version of [LLaVA Instruct Mix](https://huggingface.co/datasets/theblackcat102/llava-instruct-mix).

## Data Structure

- **Format**: [Conversational](https://huggingface.co/docs/trl/main/dataset_formats#conversational)
- **Type**: [Language-modeling](https://huggingface.co/docs/trl/main/dataset_formats#language-modeling)

Columns:
- `"images"`: The image associated with the text.
- `"prompt"`: A list of messages that form the context for the conversation.
- `"completion"`: The last message in the conversation, which is the model's response.

This structure allows models to learn from the context of the conversation, enhancing their understanding of how to generate descriptive text based on visual inputs.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/llava_instruct_mix.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("theblackcat102/llava-instruct-mix", split="train", num_proc=script_args.dataset_num_proc)

    dataset = dataset.map(
        process_example, remove_columns=["conversations", "image"], num_proc=script_args.dataset_num_proc
    )
    dataset = dataset.filter(filter_long_examples, num_proc=script_args.dataset_num_proc)
    dataset = dataset.map(split_prompt_completion, remove_columns=["messages"], num_proc=script_args.dataset_num_proc)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id, num_proc=script_args.dataset_num_proc)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
