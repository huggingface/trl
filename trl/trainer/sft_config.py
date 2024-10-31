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
from typing import Any, Dict, Optional

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`SFTTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the text field of the dataset. If provided, the trainer will automatically create a
            [`ConstantLengthDataset`] based on `dataset_text_field`.
        packing (`bool`, *optional*, defaults to `False`):
            Controls whether the [`ConstantLengthDataset`] packs the sequences of the dataset.
        learning_rate (`float`, *optional*, defaults to `2e-5`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of [`~transformers.TrainingArguments`].
        max_seq_length (`Optional[int]`, *optional*, defaults to `None`):
            Maximum sequence length for the [`ConstantLengthDataset`] and for automatically creating the dataset. If
            `None`, it uses the smaller value between `tokenizer.model_max_length` and `1024`.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset. Only used when `packing=False`.
        dataset_batch_size (`Union[int, None]`, *optional*, defaults to `1000`):
            Number of examples to tokenize per batch. If `dataset_batch_size <= 0` or `dataset_batch_size is None`,
            tokenizes the full dataset as a single batch.
        model_init_kwargs (`Optional[Dict[str, Any]]`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
        dataset_kwargs (`Optional[Dict[str, Any]]`, *optional*, defaults to `None`):
            Dictionary of optional keyword arguments to pass when creating packed or non-packed datasets.
        eval_packing (`Optional[bool]`, *optional*, defaults to `None`):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of sequences to use for the [`ConstantLengthDataset`].
        chars_per_token (`float`, *optional*, defaults to `3.6`):
            Number of characters per token to use for the [`ConstantLengthDataset`]. See
            [chars_token_ratio](https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53) for more details.
        use_liger (`bool`, *optional*, defaults to `False`):
            Monkey patch the model with Liger kernels to increase throughput and reduce memory usage.
    """

    dataset_text_field: str = "text"
    packing: bool = False
    learning_rate: float = 2.0e-5
    max_seq_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    dataset_batch_size: int = 1000
    model_init_kwargs: Optional[Dict[str, Any]] = None
    dataset_kwargs: Optional[Dict[str, Any]] = None
    eval_packing: Optional[bool] = None
    num_of_sequences: int = 1024
    chars_per_token: float = 3.6
    use_liger: bool = False
