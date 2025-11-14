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
        repo_id (`str`, *optional*, defaults to `"trl-lib/DeepMath-103K"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int`, *optional*):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/DeepMath-103K",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def process_example(example):
    solution = example["final_answer"]
    if solution not in ["True", "False", "Yes", "No"]:
        solution = f"${solution}$"
    prompt = [{"role": "user", "content": example["question"]}]
    return {"prompt": prompt, "solution": solution}


model_card = ModelCard("""
---
tags: [trl]
---

# DeepMath-103K Dataset

## Summary

[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) is meticulously curated to push the boundaries of mathematical reasoning in language models.

## Data Structure

- **Format**: [Conversational](https://huggingface.co/docs/trl/main/dataset_formats#conversational)
- **Type**: [Prompt-only](https://huggingface.co/docs/trl/main/dataset_formats#prompt-only)

Column:
- `"prompt"`: The input question.
- `"solution"`: The solution to the math problem.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/deepmath_103k.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("zwhe99/DeepMath-103K", split="train")

    dataset = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        num_proc=script_args.dataset_num_proc,
    )
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
