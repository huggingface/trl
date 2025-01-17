from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class RLOOv3Config(TrainingArguments):
    # base
    reward_model_path: Optional[str] = None
    separate_reward_tokenizer: bool = True
    judge: Optional[str] = None
    max_new_tokens: int = 64
    temperature: float = 0.9
    missing_eos_penalty: Optional[float] = None
    dataset_num_proc: Optional[int] = None
    disable_dropout: bool = True

    # preprocessing
    dataset_text_field: str = field(
        default="prompt",
        metadata={
            "help": "Name of the text field of the dataset. If provided, the trainer will automatically create a "
            "`ConstantLengthDataset` based on `dataset_text_field`."
        },
    )
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default_factory=dict,
        metadata={
            "help": "Dictionary of optional keyword arguments to pass when creating packed or non-packed datasets."
        },
    )

    # rloo
    num_mini_batches: int = field(
        default=1,
        metadata={"help": "Number of minibatches to split a batch into."},
    )
    num_ppo_epochs: int = field(
        default=1,
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
