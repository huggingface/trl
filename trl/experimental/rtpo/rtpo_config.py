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

from ...trainer.grpo_config import GRPOConfig


@dataclass
class RTPOConfig(GRPOConfig):
    """
    Configuration class for PAPOTrainer.

    PAPO (Perception-Aware Policy Optimization) extends GRPO/DAPO for multimodal reasoning by adding an implicit
    perception loss and double entropy regularization.

    Args:
        schedule_type (`str`, *optional*, defaults to `"linear"`):
            Choose a schedule type for AnnealingScheduler to control thinking guidance length. Supports: `"linear"`,
            `"cosine"`, `"exponential"`, `"piecewise"`, `"constant"`.
        direction (`str`, *optional*, defaults to `"down"`):
            Direction of the annealing schedule.
            - `"down"`: Schedule value goes from 1.0 → 0.0
            - `"up"`: Schedule value goes from 0.0 → 1.0
            Supports: `"up"`, `"down"`
        decay_rate (`float`, *optional*, defaults to `5.0`):
            The decay rate used when `schedule_type` is set to `"exponential"`. Higher values result in faster decay.
        milestones (`list[float]`, *optional*, defaults to `[0.3, 0.6, 0.9]`):
            Milestones (progress points between 0 and 1) for piecewise linear schedule. Only used when `schedule_type`
            is set to `"piecewise"`. Must be in ascending order and within [0, 1] range.
        values (`list[float]`, *optional*, defaults to `[0.2, 0.5, 0.8, 1.0]`):
            Schedule values corresponding to the milestones and boundaries. Only used when `schedule_type` is set to
            `"piecewise"`. Length must be `len(milestones) + 1`. For `direction="down"`, values typically decrease; for
            `direction="up"`, values typically increase.
        value (`float`, *optional*, defaults to `1.0`):
            Constant value for constant schedule. Only used when `schedule_type` is set to `"constant"`.
    """

    schedule_type: str = field(
        default="linear",
        metadata={
            "help": "Choose a schedule type for AnnealingScheduler to control thinking guidance length. Supports: "
            "'linear', 'cosine', 'exponential', 'piecewise', 'constant'",
        },
    )
    direction: str = field(
        default="down",
        metadata={
            "help": "Direction of the annealing schedule. 'down': Schedule value goes from 1.0 → 0.0; 'up': Schedule "
            "value goes from 0.0 → 1.0. Supports: 'up', 'down'",
        },
    )
    decay_rate: float = field(
        default=5.0,
        metadata={
            "help": "The decay rate used when schedule_type is set to 'exponential'. Higher values result in faster "
            "decay.",
        },
    )
    milestones: list[float] = field(
        default_factory=lambda: [0.3, 0.6, 0.9],
        metadata={
            "help": "Milestones (progress points between 0 and 1) for piecewise linear schedule. Only used when "
            "schedule_type is set to 'piecewise'. Must be in ascending order and within [0, 1] range.",
        },
    )
    values: list[float] = field(
        default_factory=lambda: [0.2, 0.5, 0.8, 1.0],
        metadata={
            "help": "Schedule values corresponding to the milestones and boundaries. Only used when schedule_type is "
            "set to 'piecewise'. Length must be len(milestones) + 1. For direction='down', values typically "
            "decrease; for direction='up', values typically increase.",
        },
    )
    value: float = field(
        default=1.0,
        metadata={
            "help": "Constant value for constant schedule. Only used when schedule_type is set to 'constant'.",
        },
    )
