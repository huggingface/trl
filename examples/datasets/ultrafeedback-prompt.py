# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

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
        repo_id (`str`, *optional*, defaults to `"trl-lib/ultrafeedback-prompt"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/ultrafeedback-prompt",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def to_unpaired_preference(example):
    prompt = [{"role": "user", "content": example["instruction"]}]
    return {"prompt": prompt}


def drop_long_prompt(example):
    if len(example["prompt"][0]["content"]) > 512:
        return False
    else:
        return True


model_card = ModelCard("""
---
tags: [trl]
---

# UltraFeedback - Prompts Dataset

## Summary

The UltraFeedback - Prompts dataset is a processed version of the [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset for model evaluation on specific aspects like helpfulness, honesty, and instruction-following.

## Data Structure

- **Format**: [Conversational](https://huggingface.co/docs/trl/main/dataset_formats#conversational)
- **Type**: [Prompt-only](https://huggingface.co/docs/trl/main/dataset_formats#prompt-only)

Column:
- `"prompt"`: The input question or instruction provided to the model.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/ultrafeedback-prompt.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("openbmb/UltraFeedback", split="train")

    dataset = dataset.map(
        to_unpaired_preference,
        remove_columns=["source", "instruction", "models", "completions", "correct_answers", "incorrect_answers"],
        num_proc=script_args.dataset_num_proc,
    )
    dataset = dataset.filter(drop_long_prompt)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
