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

from datasets import features, load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/rlaif-v"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = False
    repo_id: str = "trl-lib/rlaif-v"
    dataset_num_proc: Optional[int] = None


def to_conversational(example):
    """
    Convert prompt from "xxx" to [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "xxx"}]}]
    and chosen and rejected from "xxx" to [{"role": "assistant", "content": [{"type": "text", "text": "xxx"}]}].
    Images are wrapped into a list.
    """
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
    chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
    rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
    return {"prompt": prompt, "images": [example["image"]], "chosen": chosen, "rejected": rejected}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
    dataset = dataset.map(
        to_conversational,
        num_proc=script_args.dataset_num_proc,
        remove_columns=dataset.column_names,
        writer_batch_size=128,
    )

    # Cast the images to Sequence[Image] to avoid bytes format
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)

    dataset = dataset.train_test_split(test_size=0.01, writer_batch_size=128)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
