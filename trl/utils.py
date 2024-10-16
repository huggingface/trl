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

from datasets import load_dataset,Dataset,DatasetDict
import json

@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    dataset_name (`str`):
        Dataset name.
    dataset_train_split (`str`, *optional*, defaults to `"train"`):
        Dataset split to use for training.
    dataset_test_split (`str`, *optional*, defaults to `"test"`):
        Dataset split to use for evaluation.
    config (`str` or `None`, *optional*, defaults to `None`):
        Path to the optional config file.
    mixer_config (`str` or `None`, *optional*, defaults to `None`):
        Path to the optional data mixer config file.
    gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
        Whether to apply `use_reentrant` for gradient_checkpointing.
    ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
        Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar type,
        inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    config: Optional[str] = None
    mixer_config: Optional[str] = None
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False


#uses json file to create a mixed dataset
def data_mixer_from_json(json_path: str):
    # Load the JSON config file
    with open(json_path) as f:
        config = json.load(f)

    # Initialize a dictionary to hold the sampled datasets for each split
    sampled_datasets_dict = {}

    # Iterate over each split in the config
    for split, configs in config.items():
        sampled_datasets = []
        # Iterate over each dataset config in the split
        for config in configs:
            path, name, split_name, column, proportion = config
            # Load the dataset and sample the required proportion of examples
            dataset = load_dataset(path=path, name=name, split=split_name)
            num_samples = int(len(dataset) * proportion)
            dataset_slice = dataset.select(range(num_samples))
            column_data = dataset_slice[column]

            sampled_datasets.extend(column_data)
        # Combine the sampled datasets into a single dataset
        combined_dataset = Dataset.from_dict({"text": sampled_datasets}).shuffle(seed=42)
        sampled_datasets_dict[split] = combined_dataset

    # Wrap the datasets in a DatasetDict with the appropriate splits
    split_dataset = DatasetDict(sampled_datasets_dict)

    return split_dataset