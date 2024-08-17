import os
from dataclasses import dataclass

from ..trainer.utils import OnPolicyConfig


@dataclass
class PPOv2Config(OnPolicyConfig):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""

    # ppo config
    num_ppo_epochs: int = 4
    """the number of epochs to train"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""
    cliprange: float = 0.2
    """the clip range"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
