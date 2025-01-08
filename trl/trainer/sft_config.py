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

import warnings
from dataclasses import dataclass
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`SFTTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        model_init_kwargs (`Optional[dict[str, Any]]`, *optional*, defaults to `None`):
            When you provide an `str` as `model` to the [`SFTTrainer`], it will load the model with the
            `AutoModelForCausalLM.from_pretrained` method and will pass the `model_init_kwargs` as its kwargs.
        use_liger (`bool`, *optional*, defaults to `False`):
            Whether to apply Liger kernel optimizations for improved throughput and reduced memory usage.
        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column containing text data in the dataset.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        packing (`bool`, *optional*, defaults to `False`):
            If `True`, sequences in the dataset will be packed to the length specified by `max_length`. Read more about
            packing in the documentation.
        eval_packing (`Optional[bool]`, *optional*, defaults to `None`):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum sequence length. Sequences are truncated to this length if `packing` is `False`. If `packing` is
            `True`, sequences are packed to this length but not truncated.
        learning_rate (`float`, *optional*, defaults to `2e-5`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
    """

    # Model parameters
    model_init_kwargs: Optional[dict[str, Any]] = None
    use_liger: bool = False

    # Data preparation parameters
    dataset_text_field: str = "text"
    dataset_num_proc: Optional[int] = None
    packing: bool = False
    eval_packing: Optional[bool] = None
    max_length: Optional[int] = None
    
    # Training parameters
    learning_rate: float = 2e-5

    # Deprecated parameters
    dataset_kwargs: Optional[dict[str, Any]] = None
    dataset_batch_size: Optional[int] = None
    num_of_sequences: Optional[int] = None
    chars_per_token: Optional[float] = None
    max_seq_length: Optional[int] = None

    def __post_init__(self):
        if self.dataset_kwargs is not None:
            warnings.warn(
                "The `dataset_kwargs` argument is deprecated and will be removed in version 0.16.0. "
                "If you want to skip the dataset preparation, set `skip_prepare_dataset=True`.",
                DeprecationWarning,
            )
        if self.dataset_batch_size is not None:
            warnings.warn(
                "The `dataset_batch_size` argument is deprecated and will be removed in version 0.16.0. "
                "If you want to speed up the dataset preparation, you may set `dataset_num_proc`",
                DeprecationWarning,
            )
        if self.num_of_sequences is not None:
            warnings.warn(
                "The `num_of_sequences` argument is deprecated and will be removed in version 0.16.0. "
                "If you want to set the number of training steps, use the argument `max_steps` instead.",
                DeprecationWarning,
            )
        if self.chars_per_token is not None:
            warnings.warn(
                "The `chars_per_token` argument is deprecated and will be removed in version 0.16.0. "
                "If you want to set the number of training steps, use the argument `max_steps` instead.",
            )
        if self.max_seq_length is not None:
            warnings.warn(
                "The `max_seq_length` argument is deprecated and will be removed in version 0.16.0. "
                "Use `max_length` instead.",
                DeprecationWarning,
            )
            if self.max_length is None:
                self.max_length = self.max_seq_length
        return super().__post_init__()
