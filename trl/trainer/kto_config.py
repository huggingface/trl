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
from typing import Dict, Literal, Optional

from transformers import TrainingArguments

from ..import_utils import is_sklearn_available


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
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string.
        ref_model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string.
        dataset_num_proc: (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the datasets.
        loss_type: (`Literal["kto", "bco"]`, *optional*):
            The type of loss to use. Either `"kto"` the default KTO loss, `"bco"` loss from [BCO](https://arxiv.org/abs/2404.04656) paper.
        prompt_sample_size: (`int`, defaults to 1024):
            Number of prompts that are fed to density ratio classifier.
        min_density_ratio: (`float`, defaults to 0.5):
            The minimum value of the density ratio. The estimated density ratio is clamped to this value.
        max_density_ratio: (`float`, defaults to 10.0):
            The maximum value of the density ratio. The estimated density ratio is clamped to this value.
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

    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None
    precompute_ref_log_probs: bool = False
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    dataset_num_proc: Optional[int] = None

    loss_type: Literal["kto", "bco"] = "kto"

    # BCO config
    prompt_sample_size: int = 1024
    min_density_ratio: float = 0.5
    max_density_ratio: float = 10.0

    def __post_init__(self):
        super().__post_init__()

        if self.loss_type == "bco" and not is_sklearn_available():
            raise ImportError(
                "You need to install scikit-learn to use loss_type='bco' "
                "You can install it with `pip install scikit-learn`."
            )
