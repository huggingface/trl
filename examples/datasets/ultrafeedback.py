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
        model_name (`str`, *optional*, defaults to `"gpt-3.5-turbo"`):
            Language model to target. Possible values are:

                - `"alpaca-7b"`
                - `"bard"`
                - `"falcon-40b-instruct"`
                - `"gpt-3.5-turbo"` (default)
                - `"gpt-4"`
                - `"llama-2-13b-chat"`
                - `"llama-2-70b-chat"`
                - `"llama-2-7b-chat"`
                - `"mpt-30b-chat"`
                - `"pythia-12b"`
                - `"starchat"`
                - `"ultralm-13b"`
                - `"ultralm-65b"`
                - `"vicuna-33b"`
                - `"wizardlm-13b"`
                - `"wizardlm-70b"`
                - `"wizardlm-7b"`

        aspect (`str`, *optional*, defaults to `"helpfulness"`):
            Aspect to target. Possible values are:

                - `"helpfulness"` (default)
                - `"honesty"`
                - `"instruction-following"`
                - `"truthfulness"`

        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of workers to use for dataset processing.
    """

    model_name: str = "gpt-3.5-turbo"
    aspect: str = "helpfulness"
    push_to_hub: bool = False
    repo_id: str = "trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness"
    dataset_num_proc: Optional[int] = None


def to_unpaired_preference(example, model_name, aspect):
    prompt = [{"role": "user", "content": example["instruction"]}]
    model_index = example["models"].index(model_name)
    response_content = example["completions"][model_index]["response"]
    completion = [{"role": "assistant", "content": response_content}]
    score = int(example["completions"][model_index]["annotations"][aspect]["Rating"])
    label = score >= 5
    return {"prompt": prompt, "completion": completion, "label": label}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("openbmb/UltraFeedback", split="train")

    dataset = dataset.filter(
        lambda example: script_args.model_name in example["models"],
        batched=False,
        num_proc=script_args.dataset_num_proc,
    )
    dataset = dataset.map(
        to_unpaired_preference,
        remove_columns=["source", "instruction", "models", "completions", "correct_answers", "incorrect_answers"],
        fn_kwargs={"model_name": script_args.model_name, "aspect": script_args.aspect},
        num_proc=script_args.dataset_num_proc,
    )
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
