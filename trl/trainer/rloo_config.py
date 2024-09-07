import os
from dataclasses import dataclass

from ..trainer.utils import OnPolicyConfig


@dataclass
class RLOOConfig(OnPolicyConfig):
    r"""
    Configuration class for the [`RLOOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[: -len(".py")]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        rloo_k (`int`, *optional*, defaults to `2`):
            REINFORCE Leave-One-Out (RLOO) number of online samples per prompt.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    reward_model_path: str = "EleutherAI/pythia-160m"
    num_ppo_epochs: int = 4
    whiten_rewards: bool = False
    kl_coef: float = 0.05
    cliprange: float = 0.2
    rloo_k: int = 2
