import os
from dataclasses import dataclass
from typing import Literal, Optional

from trl.trainer.utils import OnPolicyConfig


@dataclass
class OnlineDPOConfig(OnPolicyConfig):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    reward_model_path: Optional[str] = None
    """the path to the reward model"""
    judge: Optional[str] = None

    num_epochs: int = 4
    """the number of epochs to train"""

    # DPO stuff w/o max_length which is included in RLOOConfig
    num_generation_per_prompt: int = 2
    """the number of generations per prompt (currently only support 2)"""
    beta: float = 0.05
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    disable_dropout: bool = True
