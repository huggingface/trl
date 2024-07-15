import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from trl.trainer.utils import OnPolicyConfig


@dataclass
class OnlineDPOConfig(OnPolicyConfig):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""

    # ppo config
    num_ppo_epochs: int = 4
    """the number of epochs to train"""

    # DPO stuff w/o max_length which is included in RLOOConfig
    num_generation_per_prompt: int = 2
    """the number of generations per prompt (currently only support 2)"""
    beta: float = 0.05
    label_smoothing: float = 0
    loss_type: Literal[
        "sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "sppo_hard", "nca_pair", "robust"
    ] = "sigmoid"
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    # max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_target_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None
