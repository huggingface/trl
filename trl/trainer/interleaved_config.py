from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments

from .grpo_config import GRPOConfig
from .sft_config import SFTConfig


@dataclass
class InterleaveConfig(TrainingArguments):
    """
    Configuration class for interleaved SFT and GRPO training.
    
    This configuration combines parameters from both SFT and GRPO trainers, allowing for
    alternating training between the two approaches. The trainer will switch between SFT
    and GRPO training every epoch.

    Parameters:
        > Parameters that control the model and reference model
        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for model initialization, used for both SFT and GRPO.
        
        > Parameters that control the training strategy
        start_with_sft (`bool`, *optional*, defaults to `True`):
            Whether to start training with SFT (True) or GRPO (False).
        sft_weight (`float`, *optional*, defaults to `0.5`):
            Weight for SFT loss when combining with GRPO loss. GRPO weight will be (1 - sft_weight).
        
        > Parameters inherited from SFT and GRPO configs
        All parameters from SFTConfig and GRPOConfig are available and will be used in their
        respective training phases.
    """
    
    # Parameters that control the model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for model initialization, used for both SFT and GRPO."
        },
    )

    # Parameters that control the training strategy
    start_with_sft: bool = field(
        default=True,
        metadata={
            "help": "Whether to start training with SFT (True) or GRPO (False)."
        },
    )
    
    sft_weight: float = field(
        default=0.5,
        metadata={
            "help": "Weight for SFT loss when combining with GRPO loss. GRPO weight will be (1 - sft_weight)."
        },
    )

    # Inherit all parameters from both SFT and GRPO configs
    sft_config: SFTConfig = field(
        default_factory=SFTConfig,
        metadata={
            "help": "Configuration for SFT training phase."
        },
    )
    
    grpo_config: GRPOConfig = field(
        default_factory=GRPOConfig,
        metadata={
            "help": "Configuration for GRPO training phase."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        
        # Validate weights
        if not 0 <= self.sft_weight <= 1:
            raise ValueError("sft_weight must be between 0 and 1") 