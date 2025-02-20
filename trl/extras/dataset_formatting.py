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

import logging
from typing import Callable, Literal, Optional, Union

from datasets import Dataset, Value
from transformers import AutoTokenizer

from ..trainer.utils import ConstantLengthDataset


FORMAT_MAPPING = {
    "chatml": [{"content": Value(dtype="string", id=None), "role": Value(dtype="string", id=None)}],
    "instruction": {"completion": Value(dtype="string", id=None), "prompt": Value(dtype="string", id=None)},
}


def conversations_formatting_function(
    tokenizer: AutoTokenizer, messages_field: Literal["messages", "conversations"], tools: Optional[list] = None
):
    r"""
    return a callable function that takes in a "messages" dataset and returns a formatted dataset, based on the tokenizer
    apply chat template to the dataset along with the schema of the list of functions in the tools list.
    """

    def format_dataset(examples):
        if isinstance(examples[messages_field][0], list):
            output_texts = []
            for i in range(len(examples[messages_field])):
                output_texts.append(
                    tokenizer.apply_chat_template(examples[messages_field][i], tokenize=False, tools=tools)
                )
            return output_texts
        else:
            return tokenizer.apply_chat_template(examples[messages_field], tokenize=False, tools=tools)

    return format_dataset


def instructions_formatting_function(tokenizer: AutoTokenizer):
    r"""
    return a callable function that takes in an "instructions" dataset and returns a formatted dataset, based on the tokenizer
    apply chat template to the dataset
    """

    def format_dataset(examples):
        if isinstance(examples["prompt"], list):
            output_texts = []
            for i in range(len(examples["prompt"])):
                converted_sample = [
                    {"role": "user", "content": examples["prompt"][i]},
                    {"role": "assistant", "content": examples["completion"][i]},
                ]
                output_texts.append(tokenizer.apply_chat_template(converted_sample, tokenize=False))
            return output_texts
        else:
            converted_sample = [
                {"role": "user", "content": examples["prompt"]},
                {"role": "assistant", "content": examples["completion"]},
            ]
            return tokenizer.apply_chat_template(converted_sample, tokenize=False)

    return format_dataset


def get_formatting_func_from_dataset(
    dataset: Union[Dataset, ConstantLengthDataset], tokenizer: AutoTokenizer, tools: Optional[list] = None
) -> Optional[Callable]:
    r"""
    Finds the correct formatting function based on the dataset structure. Currently supported datasets are:
    - `ChatML` with [{"role": str, "content": str}]
    - `instruction` with [{"prompt": str, "completion": str}]

    Args:
        dataset (Dataset): User dataset
        tokenizer (AutoTokenizer): Tokenizer used for formatting

    Returns:
        Callable: Formatting function if the dataset format is supported else None
    """
    if isinstance(dataset, Dataset):
        if "messages" in dataset.features:
            if dataset.features["messages"] == FORMAT_MAPPING["chatml"]:
                logging.info("Formatting dataset with chatml format")
                return conversations_formatting_function(tokenizer, "messages", tools)
        if "conversations" in dataset.features:
            if dataset.features["conversations"] == FORMAT_MAPPING["chatml"]:
                logging.info("Formatting dataset with chatml format")
                return conversations_formatting_function(tokenizer, "conversations", tools)
        elif dataset.features == FORMAT_MAPPING["instruction"]:
            logging.info("Formatting dataset with instruction format")
            return instructions_formatting_function(tokenizer)

    return None
