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
        repo_id (`str`, *optional*, defaults to `"trl-lib/tldr-preference"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/tldr-preference",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def to_preference(example):
    info = example["info"]
    if example["batch"] in ["batch0_cnndm", "cnndm0", "cnndm2"]:  # CNN Daily Mail batches
        article = info["article"].replace("\n\n", "\n")
        prompt = f"TITLE: {info['title']}\n\n{article}\n\nTL;DR:"
    elif example["batch"] in [f"batch{i}" for i in range(3, 23)] + ["edit_b2_eval_test"]:  # Reddit batches
        post = info["post"].replace("\n\n", "\n")
        prompt = f"SUBREDDIT: r/{info['subreddit']}\n\nTITLE: {info['title']}\n\nPOST: {post}\n\nTL;DR:"
    else:
        raise ValueError(f"Unknown batch: {example['batch']}")

    chosen_idx = example["choice"]
    rejected_idx = 1 - chosen_idx
    chosen = example["summaries"][chosen_idx]["text"]
    rejected = example["summaries"][rejected_idx]["text"]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


model_card = ModelCard("""
---
tags: [trl]
---

# TL;DR Dataset for Preference Learning

## Summary

The TL;DR dataset is a processed version of Reddit posts, specifically curated to train models using the [TRL library](https://github.com/huggingface/trl) for preference learning and Reinforcement Learning from Human Feedback (RLHF) tasks. It leverages the common practice on Reddit where users append "TL;DR" (Too Long; Didn't Read) summaries to lengthy posts, providing a rich source of paired text data for training models to understand and generate concise summaries.

## Data Structure

- **Format**: [Standard](https://huggingface.co/docs/trl/main/dataset_formats#standard)
- **Type**: [Preference](https://huggingface.co/docs/trl/main/dataset_formats#preference)

Columns:
- `"prompt"`: The unabridged Reddit post.
- `"chosen"`: The concise "TL;DR" summary appended by the author.
- `"rejected"`: An alternative summary or response that was not selected.

This structure enables models to learn the relationship between detailed content and its abbreviated form, enhancing their summarization capabilities.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/tldr_preference.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("openai/summarize_from_feedback", "comparisons")

    dataset = dataset.map(
        to_preference,
        num_proc=script_args.dataset_num_proc,
        remove_columns=["info", "summaries", "choice", "worker", "batch", "split", "extra"],
    )

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
