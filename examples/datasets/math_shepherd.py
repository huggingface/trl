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

import re
from dataclasses import dataclass, field
from itertools import chain
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
        repo_id (`str`, *optional*, defaults to `"trl-lib/math_shepherd"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/math_shepherd",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def process_example(example):
    # Replace "ки" with "ⶻ" so that the size of the "input" matches the size of the "label"
    inputs = example["input"].replace("ки", "ⶻ")

    # Find the indices of the "ⶻ" characters (that should match with the indexes of the "+" or "-" in the label)
    indexes = [m.start() for m in re.finditer("ⶻ", inputs)]

    # Sanity that all indexes are either "+" or "-"
    assert all(example["label"][idx] in ["+", "-"] for idx in indexes)

    # Get the labels
    labels = [example["label"][idx] == "+" for idx in indexes]

    # Split the inputs into steps (caution, the first step is missing here, it is the prompt)
    steps = [inputs[i:j] for i, j in zip(chain([0], indexes), chain(indexes, [None]))]

    # Remove the last step (single ⶻ)
    steps = steps[:-1]

    # Get the prompt (first part) and completions (rest)
    prompt = steps[0]
    completions = steps[1:]

    # Remove the heading "ⶻ" and the final whitespace from the completions
    assert all(completion.startswith("ⶻ") for completion in completions)
    completions = [completion[1:].strip() for completion in completions]

    # At this point, we need to retrieve the first step from the prompt.
    # First, we handle particular cases (annotation error) where we have a first label before the end of the prompt.
    if prompt.startswith(
        (
            "Mr. Rocky",
            "Parker",
            "What is the smallest positive",
            " The Myth",
            "Let $\\mathbf{a}$",
            "Find the arithmetic",
            "Determine an ordered pair",
            "Determine the ordered pair",
            "At the Quill and Scroll stationery",
            "Round to the nearest",
            r"Calculate $\sqrt{10p}",
            r"Simplify $\sqrt{28x}",
        )
    ):
        # Some spotted datasets errors where there is an annotation in the prompt: we remove it
        labels = labels[1:]

    # Then we handle the general case: we get the first step from the prompt by looking for "Step 1:" or "step 1:" or
    # (less common) "?".
    elif "Step 1:" in prompt:
        prompt, first_step = prompt.split("Step 1:")
        first_step = "Step 1:" + first_step
        completions = [first_step.strip()] + completions
    elif "step 1:" in prompt:
        prompt, first_step = prompt.split("step 1:")
        first_step = "step 1:" + first_step
        completions = [first_step.strip()] + completions
    elif "?" in prompt:
        prompt, first_step = prompt.split("?")
        prompt = prompt + "?"
        completions = [first_step.strip()] + completions
    else:
        raise ValueError(f"Prompt can't be processed: {prompt}")

    # Strip the prompt
    prompt = prompt.strip()

    # Sanity check that the length of the completions is the same as the length of the labels
    assert len(completions) == len(labels)

    return {"prompt": prompt, "completions": completions, "labels": labels}


model_card = ModelCard("""
---
tags: [trl]
---

# Math-Shepherd Dataset

## Summary

The Math-Shepherd dataset is a processed version of [Math-Shepherd dataset](peiyi9979/Math-Shepherd), designed to train models using the [TRL library](https://github.com/huggingface/trl) for stepwise supervision tasks. It provides step-by-step solutions to mathematical problems, enabling models to learn and verify each step of a solution, thereby enhancing their reasoning capabilities.

## Data Structure

- **Format**: [Standard](https://huggingface.co/docs/trl/main/dataset_formats#standard)
- **Type**: [Stepwise supervision](https://huggingface.co/docs/trl/main/dataset_formats#stepwise-supervision)

Columns:
- `"prompt"`: The problem statement.
- `"completions"`: A list of reasoning steps generated to solve the problem.
- `"labels"`: A list of booleans or floats indicating the correctness of each corresponding reasoning step.

This structure allows models to learn the correctness of each step in a solution, facilitating improved reasoning and problem-solving abilities.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/math_shepherd.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("peiyi9979/Math-Shepherd", split="train")

    dataset = dataset.map(
        process_example,
        remove_columns=["input", "label", "task"],
        num_proc=script_args.dataset_num_proc,
    )
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
