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
from typing import Any, Dict, Optional

from transformers import TrainingArguments


@dataclass
class BCOConfig(TrainingArguments):
    r"""
    Configuration class for the [`BCOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the sequences in the batch. This argument is required if you want to use the default
            data collator.
        max_prompt_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_completion_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum length of the target. This argument is required if you want to use the default data collator
            and your model is an encoder-decoder.
        beta (`float`, *optional*, defaults to `0.1`):
            Beta factor in BCO loss. Higher beta means less divergence from the initial policy.
        label_pad_token_id (`int`,  *optional*, defaults to `-100`):
            Label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`Optional[int]`, *optional*, defaults to `None`):
            Padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use
            the default data collator.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, *optional*, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful
            if you want to train without the reference model and reduce the total GPU memory needed.
        model_init_kwargs (`Optional[Dict[str, Any]]`, *optional*, defaults to `None`):
            Dict of optional kwargs to pass when instantiating the model from a string.
        ref_model_init_kwargs (`Optional[Dict[str, Any]]`, *optional*, defaults to `None`):
            Dict of optional kwargs to pass when instantiating the reference model from a string.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the datasets.
        prompt_sample_size (`int`, *optional*, defaults to `1024`):
            Number of prompts that are fed to density ratio classifier.
        min_density_ratio (`float`, *optional*, defaults to `0.5`):
            Minimum value of the density ratio. The estimated density ratio is clamped to this value.
        max_density_ratio (`float`, *optional*, defaults to `10.0`):
            Maximum value of the density ratio. The estimated density ratio is clamped to this value.
    """

    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    beta: float = 0.1
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None
    precompute_ref_log_probs: bool = False
    model_init_kwargs: Optional[Dict[str, Any]] = None
    ref_model_init_kwargs: Optional[Dict[str, Any]] = None
    dataset_num_proc: Optional[int] = None

    # BCO config
    prompt_sample_size: int = 1024
    min_density_ratio: float = 0.5
    max_density_ratio: float = 10.0
