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

from dataclasses import dataclass, field
from typing import Optional

from .ppo_config import PPOConfig


@dataclass
class RePOConfig(PPOConfig):
    """
    Configuration class for RePO (Replay-Enhanced Policy Optimization) trainer.
    Inherits from PPOConfig.

    Args:
        replay_buffer_size (`int`, *optional*, defaults to `10000`):
            Maximum capacity of the replay buffer.
        replay_batch_size (`int`, *optional*, defaults to `32`):
            Number of samples to draw from the replay buffer for each training step.
        kl_coef_replay (`float`, *optional*, defaults to `0.1`):
            Coefficient for the KL divergence term with off-policy data. This is used to regularize
            the policy update towards the reference policy using samples from the replay buffer.
        importance_sampling_clip (`float`, *optional*, defaults to `10.0`):
            Clipping value for importance sampling weights (rho). This helps to stabilize training
            by preventing excessively large importance weights.
        replay_warmup_steps (`int`, *optional*, defaults to `100`):
            Number of training steps to perform before starting to sample from the replay buffer.
            This allows the buffer to be populated with some initial experiences.
        use_replay_buffer (`bool`, *optional*, defaults to `True`):
            Whether to use the replay buffer and incorporate off-policy updates. If False,
            the trainer will behave like a standard PPO trainer.
    """

    replay_buffer_size: int = 10000
    replay_batch_size: int = 32
    kl_coef_replay: float = 0.1
    importance_sampling_clip: float = 10.0
    replay_warmup_steps: int = 100
    use_replay_buffer: bool = True
