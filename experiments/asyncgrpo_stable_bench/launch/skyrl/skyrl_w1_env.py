"""W1 math env for SkyRL: single turn, reward = workloads/math/reward.py::compute_score (math_verify on the last \\boxed{})."""

from typing import Any

from reward_safe import compute_score  # workloads/math/reward_safe.py (SHARED_W1 rev. 7), on PYTHONPATH
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class W1MathEnv(BaseTextEnv):
    def __init__(self, env_config: Any = None, extras: dict[str, Any] = {}):
        super().__init__()
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        return BaseTextEnvStepOutput(
            observations=[], reward=compute_score(action, self.ground_truth), done=True, metadata={}
        )
