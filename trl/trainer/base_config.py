# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class BaseConfig(TrainingArguments):
    """
    Base configuration class for all TRL trainer configurations.

    Subclasses [`~transformers.TrainingArguments`] and overrides fields that are common across TRL trainers or that
    contain unescaped "%" characters which would cause argparse to raise a `TypeError` when rendering `--help` output.

    Parameters:
        logging_steps (`float`, *optional*, defaults to `10`):
            Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in
            range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            Whether to enable gradient checkpointing to trade compute for memory. Reduces memory usage by clearing
            activations during forward pass and recomputing them during backward pass. Enables training larger models
            or batch sizes at the cost of ~20% slower training.
        bf16 (`bool`, *optional*):
            Whether to use bfloat16 (BF16) mixed precision instead of 32-bit. Generally preferred over FP16 due to
            better numerical stability and no loss scaling required. Requires Ampere or higher NVIDIA architecture or
            Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if `fp16` is not set.
        lr_scheduler_kwargs (`dict` or `str`, *optional*):
            Additional parameters for the lr_scheduler, such as `{'num_cycles': 1}` for cosine with hard restarts. See
            the documentation of each scheduler for possible values.
        use_liger_kernel (`bool`, *optional*, defaults to `False`):
            Enable [Liger Kernel](https://github.com/linkedin/Liger-Kernel) optimizations. Increases multi-GPU
            throughput by ~20% and reduces memory usage by ~60%. Works with Flash Attention, FSDP, and DeepSpeed.
            Currently, supports Llama, Mistral, Mixtral, and Gemma models.
        torch_empty_cache_steps (`int`, *optional*):
            Number of steps to wait before calling `torch.<device>.empty_cache()`. If left unset or set to None, cache
            will not be emptied. This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of
            about [10% slower performance](https://github.com/huggingface/transformers/issues/31372).
    """

    # Override fields from TrainingArguments to set defaults preferred by all TRL trainers.
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20%% slower training."
        },
    )
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )
    # Transformers 4.57.0 introduced a bug that caused the dtype of `lr_scheduler_kwargs` to be unparsable. This issue
    # was fixed in https://github.com/huggingface/transformers/pull/41322 and released in 4.57.5. We add a temporary
    # workaround here, which can be removed once we drop support for versions older than 4.57.5.
    lr_scheduler_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard "
            "restarts. See the documentation of each scheduler for possible values."
        },
    )

    # Override fields from TrainingArguments whose help strings contain unescaped "%" characters.
    # argparse interprets "%" as a format specifier, raising TypeError when rendering --help output.
    # Fixed upstream in transformers v5.3.0, but overridden here to support older versions.
    # - Introduced in v5.2.0; fixed in v5.3.0
    use_liger_kernel: bool = field(
        default=False,
        metadata={
            "help": "Enable Liger Kernel optimizations. Increases throughput by ~20%% and reduces memory by ~60%%."
        },
    )
    # - Introduced in v4.54.1; fixed in v5.3.0
    torch_empty_cache_steps: int | None = field(
        default=None,
        metadata={
            "help": "Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10%% slower performance. If None, cache will not be emptied."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()
