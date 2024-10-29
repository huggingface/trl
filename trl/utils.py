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
    gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
        Whether to apply `use_reentrant` for gradient_checkpointing.
    ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
        Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar type,
        inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: str
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    config: Optional[str] = None
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False
