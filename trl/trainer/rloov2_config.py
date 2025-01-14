from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class RLOOv2Config(TrainingArguments):
    # base
    reward_model_path: Optional[str] = None
    judge: Optional[str] = None
    max_new_tokens: int = 64
    temperature: float = 0.9
    missing_eos_penalty: Optional[float] = None
    dataset_num_proc: Optional[int] = None
    disable_dropout: bool = True

    # rloo
    num_ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of epochs to train."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    rloo_k: int = field(
        default=2,
        metadata={"help": "REINFORCE Leave-One-Out (RLOO) number of online samples per prompt."},
    )
    normalize_reward: bool = field(
        default=False,
        metadata={"help": "Whether to normalize rewards"},
    )
    reward_clip_range: float = field(
        default=10.0,
        metadata={"help": "Clip range for rewards"},
    )
    normalize_advantage: bool = field(
        default=False,
        metadata={"help": "Whether to normalize advantages"},
    )
    token_level_kl: bool = field(
        default=True,
        metadata={"help": "Whether to use token-level KL penalty or sequence-level KL penalty"},
    )
