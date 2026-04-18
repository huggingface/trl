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

from .grpo_config import GRPOConfig


@dataclass
class TPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`TPOTrainer`].

    This class extends [`GRPOConfig`] with defaults for Target Policy Optimization. For a full list of training
    arguments, please refer to [`~transformers.TrainingArguments`] and [`GRPOConfig`].

    Parameters:
        loss_type (`str`, *optional*, defaults to `"tpo"`):
            Loss formulation. `TPOConfig` requires this to stay set to `"tpo"`.
        tpo_target_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature used to build the TPO target distribution. Lower values make the target more concentrated on
            high-scoring completions.
    """

    loss_type: str = field(
        default="tpo",
        metadata={"help": "Loss formulation. `TPOConfig` requires this to stay set to `tpo`."},
    )

    def __post_init__(self):
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = 1

        super().__post_init__()

        if self.loss_type != "tpo":
            raise ValueError(f"TPOConfig requires loss_type='tpo'. You provided {self.loss_type!r}.")
