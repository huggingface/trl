import warnings
from .ppo_trainer import PPOTrainer

# Define an alias for PPOv2Trainer that raises a warning
class PPOv2Trainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`PPOv2Trainer` is deprecated and has been renamed to `PPOTrainer`. Please use `PPOTrainer` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
