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
class VIMPOConfig(GRPOConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`experimental.vimpo.VIMPOTrainer`].

    [`VIMPOConfig`] inherits the rollout, reward, and batching parameters from [`GRPOConfig`]. VIMPO uses the same
    online RLVR data path as GRPO, but replaces the GRPO policy loss with the Value-Implicit Policy Optimization
    objective from [VIMPO: Value-Implicit Policy Optimization for LLMs](https://huggingface.co/papers/2606.20008).

    Parameters:
        vimpo_beta (`float`, *optional*, defaults to `5e-4`):
            Coefficient beta applied to the policy-reference log-ratio and exact KL terms in the VIMPO value and actor
            objectives.
        vimpo_actor_coeff (`float`, *optional*, defaults to `5e-3`):
            Coefficient c_A applied to the PPO-style actor branch. Set to `0.0` to train with only the terminal value
            loss.
        vimpo_gae_lambda (`float`, *optional*, defaults to `1.0`):
            Lambda parameter used to accumulate the policy-implied one-step advantages before the actor loss.
        scale_rewards (`str`, *optional*, defaults to `"none"`):
            VIMPO's terminal value target uses unscaled centered rewards, `R - mean_group(R)`.
    """

    scale_rewards: str = field(
        default="none",
        metadata={"help": "VIMPO's terminal value target uses unscaled centered rewards, `R - mean_group(R)`."},
    )
    vimpo_beta: float = field(
        default=5e-4,
        metadata={"help": "Coefficient beta applied to the policy-reference log-ratio and exact KL terms."},
    )
    vimpo_actor_coeff: float = field(
        default=5e-3,
        metadata={"help": "Coefficient applied to the PPO-style actor branch. Set to 0.0 for value-only VIMPO."},
    )
    vimpo_gae_lambda: float = field(
        default=1.0,
        metadata={"help": "Lambda parameter used for VIMPO's GAE-style actor advantage accumulation."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.vimpo_beta <= 0.0:
            raise ValueError(f"vimpo_beta must be > 0.0, got {self.vimpo_beta}.")
        if self.vimpo_actor_coeff < 0.0:
            raise ValueError(f"vimpo_actor_coeff must be >= 0.0, got {self.vimpo_actor_coeff}.")
        if not 0.0 <= self.vimpo_gae_lambda <= 1.0:
            raise ValueError(f"vimpo_gae_lambda must be in [0.0, 1.0], got {self.vimpo_gae_lambda}.")
