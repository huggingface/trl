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
class GRPOWithReplayBufferConfig(GRPOConfig):
    """
    New Parameters:
        replay_buffer_size (`int`, *optional*, defaults to `0`):
                A cache that stores the rollouts with the highest advantage scores and variance per group. If a new
                group has 0 variance, it is replaced with a group sampled from the replay buffer.
    """

    replay_buffer_size: int = field(
        default=64,
        metadata={
            "help": "A cache that stores the rollouts with the highest advantage scores and variance per group. If a new group has 0 variance, it is replaced with a group sampled from the replay buffer."
        },
    )
