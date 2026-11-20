"""Snapshot of configs/openrlhf/w1_reward.py before the NOTES.md item 19 fix (reward in the calling thread), kept for reward_gil_check.py.

OpenRLHF loads this file and calls `reward_func(queries, prompts, labels)` once per rollout, in a worker thread
of the vLLM rollout actor. `queries` is prompt + completion decoded with special tokens, `prompts` the chat-templated
prompt string, `labels` the dataset `answer` column.
"""

import sys

import torch


sys.path.insert(0, "/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/math")
from reward import compute_score  # noqa: E402


def reward_func(queries, prompts, labels, **kwargs):
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels, strict=False):
        completion = query[len(prompt) :] if query.startswith(prompt) else query
        rewards.append(compute_score(completion, label))
    rewards = torch.tensor(rewards, dtype=torch.float)
    return {"rewards": rewards, "scores": rewards, "extra_logs": {}}
