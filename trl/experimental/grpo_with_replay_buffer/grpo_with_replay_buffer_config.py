from dataclasses import dataclass, field
from trl.trainer.grpo_config import GRPOConfig

@dataclass
class GRPOWithReplayBufferConfig(GRPOConfig):
    """
    New Parameters:
        replay_buffer_size (`int`, *optional*, defaults to `0`):
                A cache that stores the rollouts with the highest advantage scores and variance per group. If a new group
                has 0 variance, it is replaced with a group sampled from the replay buffer.
    """
    replay_buffer_size: int = field(
        default=64,
        metadata={
            "help": "A cache that stores the rollouts with the highest advantage scores and variance per group. If a new group has 0 variance, it is replaced with a group sampled from the replay buffer."
        },
    )