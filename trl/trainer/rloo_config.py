from dataclasses import dataclass

from ..trainer.utils import OnPolicyConfig


@dataclass
class RLOOConfig(OnPolicyConfig):
    # ppo config
    num_ppo_epochs: int = 4
    """the number of epochs to train"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""
    cliprange: float = 0.2
    """the clip range"""

    # rloo config
    rloo_k: int = 2
    """REINFORCE Leave-One-Out (RLOO) number of online samples per prompt"""
