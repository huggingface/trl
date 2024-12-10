# Copyright 2024 The HuggingFace Team. All rights reserved.
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
        repo_id (`str`, *optional*, defaults to `"trl-lib/hh-rlhf-helpful-base"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = False
    repo_id: str = "trl-lib/hh-rlhf-helpful-base"
    dataset_num_proc: Optional[int] = None


def common_start(str1: str, str2: str) -> str:
    # Zip the two strings and iterate over them together
    common_chars = []
    for c1, c2 in zip(str1, str2):
        if c1 == c2:
            common_chars.append(c1)
        else:
            break
    # Join the common characters and return as a string
    return "".join(common_chars)


def extract_dialogue(example: str) -> list[dict[str, str]]:
    # Extract the prompt, which corresponds to the common start of the chosen and rejected dialogues
    prompt_text = common_start(example["chosen"], example["rejected"])

    # The chosen and rejected may share a common start, so we need to remove the common part
    if not prompt_text.endswith("\n\nAssistant: "):
        prompt_text = prompt_text[: prompt_text.rfind("\n\nAssistant: ")] + "\n\nAssistant: "

    # Extract the chosen and rejected lines
    chosen_line = example["chosen"][len(prompt_text) :]
    rejected_line = example["rejected"][len(prompt_text) :]

    # Remove the generation prompt ("\n\nAssistant: ") from the prompt
    prompt_text = prompt_text[: -len("\n\nAssistant: ")]

    # Split the string at every occurrence of "Human: " or "Assistant: "
    prompt_lines = re.split(r"(\n\nAssistant: |\n\nHuman: )", prompt_text)

    # Remove the first element as it's empty
    prompt_lines = prompt_lines[1:]

    prompt = []
    for idx in range(0, len(prompt_lines), 2):
        role = "user" if prompt_lines[idx] == "\n\nHuman: " else "assistant"
        content = prompt_lines[idx + 1]
        prompt.append({"role": role, "content": content})

    # Remove the prompt from the chosen and rejected dialogues
    chosen = [{"role": "assistant", "content": chosen_line}]
    rejected = [{"role": "assistant", "content": rejected_line}]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    dataset = dataset.map(extract_dialogue, num_proc=script_args.dataset_num_proc)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
