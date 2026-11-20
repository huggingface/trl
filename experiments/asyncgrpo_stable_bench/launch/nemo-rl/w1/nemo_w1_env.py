"""W1 math env for NeMo-RL: single turn, reward = workloads/math/reward_safe.py::compute_score (SHARED_W1 rev. 7).

Same shape as NeMo-RL's MathEnvironment (one Ray actor, `step` scores the assistant text against the ground truth and
ends the episode), but it calls the bench's shared reward instead of NeMo-RL's own math_verify wrapper.
"""

import ray
import torch
from reward_safe import compute_score  # workloads/math on PYTHONPATH

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


@ray.remote(max_concurrency=64)
class W1MathEnv(EnvironmentInterface):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def step(self, message_log_batch, metadata):
        texts = ["".join(str(m["content"]) for m in log if m["role"] == "assistant") for log in message_log_batch]
        rewards = [compute_score(t, meta["ground_truth"]) for t, meta in zip(texts, metadata)]
        n = len(texts)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": "Environment: " + ("correct" if r else "incorrect")} for r in rewards],
            metadata=metadata,
            next_stop_strings=[None] * n,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.ones(n, dtype=torch.bool),
            answers=[None] * n,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(self, batch: BatchedDataDict):
        rewards = batch["total_reward"]
        return batch, {"accuracy": rewards.mean().item()}
