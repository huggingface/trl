# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
from dataclasses import dataclass

from ..trainer.utils import OnPolicyConfig
import os
from dataclasses import dataclass

from ..trainer.utils import OnPolicyConfig


@dataclass
class GRPOConfig(OnPolicyConfig):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""

    # ppo config
    num_grpo_epochs: int = 4
    """the number of epochs to train"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""
    cliprange: float = 0.2
    """the clip range"""

    # GRPO specific changes
    use_process_supervision : bool = False
    """Enable process Supervision"""
    use_iterative_reward_model_training : bool = True
    """Enable iterative reward model training"""
    sampling_group_size : int = 4
    """Group size sampling from the old policy model"""
    sampling_strategy : str = "top_p"
    """Sampling strategy for the group size sampling, only support top_p and top_k"""
    sampling_strategy_top_p : float = 0.95
    """Sampling strategy parameter for the group size sampling"""
    sampling_strategy_top_k : int = 0
    """Sampling strategy parameter for top k sampling"""
    sampling_temperature : float = 0.1
