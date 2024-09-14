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

from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
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
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = False
    repo_id: str = "trl-lib/tldr-preference"
    dataset_num_proc: Optional[int] = None


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


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("openai/summarize_from_feedback", "comparisons")

    dataset = dataset.map(
        to_preference,
        num_proc=args.dataset_num_proc,
        remove_columns=["info", "summaries", "choice", "worker", "batch", "split", "extra"],
    )

    if args.push_to_hub:
        dataset.push_to_hub(args.repo_id)
