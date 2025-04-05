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
        repo_id (`str`, *optional*, defaults to `"trl-lib/tldr"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/tldr",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def to_prompt_completion(example):
    tldr_format_str = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    prompt = tldr_format_str.format(subreddit=example["subreddit"], title=example["title"], post=example["post"])
    completion = " " + example["summary"]  # Add a space to separate the prompt from the completion
    return {"prompt": prompt, "completion": completion}


model_card = ModelCard("""
---
tags: [trl]
---

# TL;DR Dataset

## Summary

The TL;DR dataset is a processed version of Reddit posts, specifically curated to train models using the [TRL library](https://github.com/huggingface/trl) for summarization tasks. It leverages the common practice on Reddit where users append "TL;DR" (Too Long; Didn't Read) summaries to lengthy posts, providing a rich source of paired text data for training summarization models.

## Data Structure

- **Format**: [Standard](https://huggingface.co/docs/trl/main/dataset_formats#standard)
- **Type**: [Prompt-completion](https://huggingface.co/docs/trl/main/dataset_formats#prompt-completion)

Columns:
- `"prompt"`: The unabridged Reddit post.
- `"completion"`: The concise "TL;DR" summary appended by the author.

This structure enables models to learn the relationship between detailed content and its abbreviated form, enhancing their summarization capabilities.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/tldr.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Filtered reddit TL;DR dataset from https://github.com/openai/summarize-from-feedback?tab=readme-ov-file#reddit-tldr-dataset
    data_files = {
        "train": "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/train.jsonl",
        "validation": "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/valid.jsonl",
        "test": "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/test.jsonl",
    }
    dataset = load_dataset("json", data_files=data_files)

    dataset = dataset.map(
        to_prompt_completion,
        num_proc=script_args.dataset_num_proc,
        remove_columns=["id", "subreddit", "title", "post", "summary"],
    )

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
