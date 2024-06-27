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
from typing import Dict, Optional

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    r"""
    Initialize SFTConfig.

    Args:
        dataset_text_field (`Optional[str]`):
            The name of the text field of the dataset, in case this is passed by a user, the trainer will automatically create a
            `ConstantLengthDataset` based on the `dataset_text_field` argument. Defaults to None.
        packing (`Optional[bool]`):
            Used only in case `dataset_text_field` is passed. This argument is used by the `ConstantLengthDataset` to pack the sequences
            of the dataset. Defaults to False.
        max_seq_length (`Optional[int]`):
            The maximum sequence length to use for the `ConstantLengthDataset` and for automatically creating the Dataset. Defaults to min of the smaller of the `tokenizer.model_max_length` and `1024`.
        dataset_num_proc (`Optional[int]`):
            The number of workers to use to tokenize the data. Only used when `packing=False`. Defaults to None.
        dataset_batch_size (`int`):
            The number of examples to tokenize per batch. If batch_size <= 0 or batch_size == None,
            tokenize the full dataset as a single batch. Defaults to 1000.
        neftune_noise_alpha (`Optional[float]`):
            If not `None`, this will activate NEFTune noise embeddings. This has been proven to drastically improve model performances for instruction
            fine-tuning. Check out the original paper here: https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune
        model_init_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string.
        dataset_kwargs: (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when creating packed or non-packed datasets
        eval_packing: (`Optional[bool]`, *optional*):
            Whether to pack the eval dataset as well. Defaults to `packing` if `None` is passed.
        num_of_sequences (`Optional[int]`):
            The number of sequences to use for the `ConstantLengthDataset`. Defaults to `1024`.
        chars_per_token (`Optional[float]`):
            The number of characters per token to use for the `ConstantLengthDataset`. Defaults to `3.6`. You can check how this is computed in the
            stack-llama example:
            [chars_token_ratio](https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53).
    """

    dataset_text_field: Optional[str] = None
    packing: Optional[bool] = False
    max_seq_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    dataset_batch_size: int = 1000
    neftune_noise_alpha: Optional[float] = None
    model_init_kwargs: Optional[Dict] = None
    dataset_kwargs: Optional[Dict] = None
    eval_packing: Optional[bool] = None
    num_of_sequences: Optional[int] = 1024
    chars_per_token: Optional[float] = 3.6
