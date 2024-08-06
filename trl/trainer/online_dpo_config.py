import os
from dataclasses import dataclass
from typing import Literal

from trl.trainer.utils import OnPolicyConfig


@dataclass
class OnlineDPOConfig(OnPolicyConfig):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""

    num_epochs: int = 4
    """the number of epochs to train"""

    beta: float = 0.05
    """the entropy regularization coefficient of DPO"""
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    """the type of loss to use for online DPO"""
    disable_dropout: bool = True
    """whether to disable dropout of the model during training"""
