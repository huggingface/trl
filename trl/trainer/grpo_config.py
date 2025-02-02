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

import warnings
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_init_kwargs (`dict`, *optional*, defaults to {"device": "auto", "gpu_memory_utilization": 0.9})
            "Keyword arguments for `vllm.LLM.__init__` when `use_vllm` is true"

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        per_device_train_batch_size (`int`, *optional*, defaults to `1`):
            Number of prompts sampled per device for training. The actual batch passed into the model will be this
            value multiplied by `num_generations`.
        gradient_accumulation_steps (`int`, *optional*, defaults to `8`):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={"help": "Number of generations to sample."},
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )
    vllm_init_kwargs: Optional[dict] = field(
        default_factory=lambda: {
            "device": "auto",
            "gpu_memory_utilization": 0.9,
        },
        metadata={
            "help": "Keyword arguments for `vllm.LLM.__init__` when `use_vllm` is true"
        },
    ) 
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Deprecated. Set `device` in `vllm_init_kwargs` instead."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Deprecated. Set `gpu_memory_utilization` in `vllm_init_kwargs` instead." 
        },
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    # GRPO generates multiple completions per prompt, increasing memory usage.
    # To accommodate this, the per-device train batch size is decreased (overriden from the parent class),
    # and the number gradient accumulation steps is increased to maintain the effective batch size.
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "Number of prompts sampled per device for training. The actual batch passed into the model will "
            "be this value multiplied by `num_generations`."
        },
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={
            "help": "Number of updates steps to accumulate the gradients for, before performing a backward/update "
            "pass."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.vllm_device:
            warnings.warn(
                "`vllm_device` is deprecated. Set `device` in `vllm_init_kwargs` instead.",
                DeprecationWarning,
            )
            self.vllm_init_kwargs["device"] = self.vllm_device

        if self.vllm_gpu_memory_utilization:
            warnings.warn(
                "`vllm_gpu_memory_utilization` is deprecated. Set `gpu_memory_utilization` in `vllm_init_kwargs` instead.",
                DeprecationWarning,
            )
            self.vllm_init_kwargs["gpu_memory_utilization"] = self.vllm_gpu_memory_utilization
