# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from typing import Any

from transformers import TrainingArguments

from ...trainer.grpo_config import GRPOConfig


@dataclass
class MiniLLMConfig(GRPOConfig):
    """
    Configuration class for [`MiniLLMTrainer`].

    This class includes only the parameters that are specific to MiniLLM training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] and [`GRPOConfig`] documentation.

    Args:
        teacher_model_init_kwargs (`dict[str, Any]]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        rkl_advantage (`bool`, *optional*, defaults to `True`):
            Whether to add the reverse KL advantage to the reward advantage.
        single_step_decomposition (`bool`, *optional*, defaults to `True`):
            Whether to use single-step decomposition for the KL divergence computation.
        kd_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for knowledge distillation. Higher temperatures produce softer probability distributions over
            classes.
        gamma (`float`, *optional*, defaults to `0.0`):
            Discount factor for future rewards in reinforcement learning.
        length_normalization (`bool`, *optional*, defaults to `True`):
            Whether to apply length normalization to the rewards.
    """

    teacher_model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "teacher model from a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    rkl_advantage: bool = field(
        default=True,
        metadata={"help": "Whether to add the reverse KL advantage to the reward advantage."},
    )
    single_step_decomposition: bool = field(
        default=True,
        metadata={"help": "Whether to use single-step decomposition for the KL divergence computation."},
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for knowledge distillation. Higher temperatures produce softer probability "
            "distributions over classes."
        },
    )
    gamma: float = field(
        default=0.0,
        metadata={"help": "Discount factor for future rewards in reinforcement learning."},
    )
    length_normalization: bool = field(
        default=True,
        metadata={"help": "Whether to apply length normalization to the rewards."},
    )

    def __post_init__(self):
        # We do not use the post_init of GRPOConfig because:
        # 1. num_generations can be < 2 in MiniLLMConfig. Scale_rewards must be set to "none" to avoid nan.
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        TrainingArguments.__post_init__(self)

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)
        if self.num_generations == 1:
            self.scale_rewards = "none"

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by num_generations ({self.num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.use_liger_loss is not None:
            warnings.warn(
                "The `use_liger_loss` argument is deprecated and will be removed in version 0.28.0. Please use "
                "`use_liger_kernel` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.use_liger_kernel = self.use_liger_loss

        if self.delta is not None and self.use_liger_kernel:
            raise ValueError("Liger kernel does not support two-sided GRPO loss yet.")
