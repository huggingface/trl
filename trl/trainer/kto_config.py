# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from transformers import DataCollator, TrainingArguments


@dataclass
class KTOConfig(TrainingArguments):
    r"""
    KTOConfig collects all training arguments related to the [`KTOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, *optional*, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, *optional*, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`int`, *optional*, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        beta (`float`, defaults to 0.1):
            The beta factor in KTO loss. Higher beta means less divergence from the initial policy.
        desirable_weight (`float`, *optional*, defaults to 1.0):
            The desirable losses are weighed by this factor to counter unequal number of desirable and undesirable paris.
        undesirable_weight (`float`, *optional*, defaults to 1.0):
            The undesirable losses are weighed by this factor to counter unequal number of desirable and undesirable pairs.
        data_collator (`transformers.DataCollator`, *optional*, defaults to `None`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
    """

    max_length: Optional[int] = None
    """The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."""
    max_prompt_length: Optional[int] = None
    """The maximum length of the prompt. This argument is required if you want to use the default data collator."""
    max_completion_length: Optional[int] = None
    """The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder."""
    beta: float = 0.1
    """The beta factor in KTO loss. Higher beta means less divergence from the initial policy."""
    desirable_weight: Optional[float] = 1.0
    """The desirable losses are weighed by this factor."""
    undesirable_weight: Optional[float] = 1.0
    """The undesirable losses are weighed by this factor."""

    data_collator: Optional[DataCollator] = None
    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
