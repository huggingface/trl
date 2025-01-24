from dataclasses import dataclass, field
from typing import Optional, Union
from .grpo_config import GRPOConfig

@dataclass
class PrimeConfig(GRPOConfig):
    """
    Configuration class for the PrimeTrainer.
    Extends GRPOConfig with PRIME-specific parameters.
    """
    
    # Reward model parameters
    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model or its name on the Hub"},
    )
    
    reward_model_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Additional kwargs for reward model initialization"},
    )
    
    # PRIME specific parameters
    prime_granularity: str = field(
        default="token",
        metadata={"help": "Granularity of process rewards: 'token' or 'whole'"},
    )
    
    prime_norm: str = field(
        default="batch_norm",
        metadata={"help": "Normalization method for process rewards"},
    )
    
    prime_ref_type: str = field(
        default="freeze",
        metadata={"help": "Reference model type: 'freeze' or 'policy'"},
    )
    
    prime_beta_train: float = field(
        default=0.05,
        metadata={"help": "Beta coefficient for training"},
    )
    
    reward_model_coef: float = field(
        default=0.0,
        metadata={"help": "Weight for the reward model score"},
    )