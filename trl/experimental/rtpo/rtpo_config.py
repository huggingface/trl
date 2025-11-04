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

from dataclasses import dataclass, field
from ...trainer.grpo_config import GRPOConfig


@dataclass
class RTPOConfig(GRPOConfig):
    """
    Configuration class for PAPOTrainer.

    PAPO (Perception-Aware Policy Optimization) extends GRPO/DAPO for multimodal reasoning by adding an implicit
    perception loss and double entropy regularization.

    Args:
        schedule_type (`str`,defaults to linear):
            Choose a schedule type for AnnealingScheduler to control thinking guidance length.
            Supports: linear, cosine, exponential, piecewise, constant
        direction (`str`,defaults to down):
            down: 1 -> 0, up: 0 -> 1.
            Supports: up, down
        decay_rate (`float`, defaults to 5.0):
            The decay rate used when schedule_type is set to 'exponential'.
        milestones (`list[float]`, defaults to [0.3, 0.6, 0.9]):
            Milestones for piecewise schedule.
        values (`list[float]`, defaults to [0.2, 0.5, 0.8, 1.0]):
            Values corresponding to milestones.
        value (`float`, defaults to 1.0):
            Constant value for constant schedule.
    """

    schedule_type: str = "linear"
    direction: str = "down"
    # when schedule_type set to exponential
    decay_rate: float = 5.0 if schedule_type == "exponential" else None
    # when schedule_type set to piecewise
    if schedule_type == "piecewise":
        milestones: list[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
        values: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.8, 1.0])
    else:
        milestones = values = None
    # when schedule_type set to constant
    value: float = 1.0 if schedule_type == "constant" else None
