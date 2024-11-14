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
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/lm-human-preferences-descriptiveness"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = False
    repo_id: str = "trl-lib/lm-human-preferences-descriptiveness"
    dataset_num_proc: Optional[int] = None


# Edge cases handling: remove the cases where all samples are the same
def samples_not_all_same(example):
    return not all(example["sample0"] == example[f"sample{j}"] for j in range(1, 4))


def to_prompt_completion(example, tokenizer):
    prompt = tokenizer.decode(example["query"]).strip()
    best_idx = example["best"]
    chosen = tokenizer.decode(example[f"sample{best_idx}"])
    for rejected_idx in range(4):  # take the first rejected sample that is different from the chosen one
        rejected = tokenizer.decode(example[f"sample{rejected_idx}"])
        if chosen != rejected:
            break
    assert chosen != rejected
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset(
        "json",
        data_files="https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json",
        split="train",
    )

    dataset = dataset.filter(samples_not_all_same, num_proc=script_args.dataset_num_proc)

    dataset = dataset.map(
        to_prompt_completion,
        num_proc=script_args.dataset_num_proc,
        remove_columns=["query", "sample0", "sample1", "sample2", "sample3", "best"],
        fn_kwargs={"tokenizer": AutoTokenizer.from_pretrained("gpt2")},
    )

    # train_size taken from https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/launch.py#L79)
    dataset = dataset.train_test_split(train_size=4992)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
