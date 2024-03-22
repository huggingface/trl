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
from dataclasses import dataclass
from typing import Dict, Optional

from transformers import TrainingArguments


@dataclass
class ORPOConfig(TrainingArguments):
    r"""
    ORPOConfig collects all training arguments related to the [`ORPOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`int`, defaults to `None`):
            The maximum length of the completions. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        beta (`float`, defaults to 0.1):
            The beta factor in ORPO loss (lambda/alpha in paper/code) that is the weight of the relative loss ratio in the SFT loss.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model`.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
    """

    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None

    beta: float = 0.1
    disable_dropout: bool = True

    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None

    model_init_kwargs: Optional[Dict] = None

    dataset_num_proc: Optional[int] = None
