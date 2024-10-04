import warnings
from .ppo_config import PPOConfig

# Define an alias for PPOv2Trainer that raises a warning
class PPOv2Config(PPOConfig):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`PPOv2Config` is deprecated and has been renamed to `PPOConfig`. Please use `PPOConfig` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
