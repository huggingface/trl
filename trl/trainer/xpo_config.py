from dataclasses import dataclass

from trl.trainer.online_dpo_config import OnlineDPOConfig


@dataclass
class XPOConfig(OnlineDPOConfig):
    alpha: float = 0.05
