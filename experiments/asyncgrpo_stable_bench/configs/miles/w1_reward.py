"""W1 reward for miles: `--custom-rm-path w1_reward.reward` (configs/miles and workloads/math on PYTHONPATH).

miles calls `await reward(args, sample)` once per sample (miles/rollout/rm_hub/__init__.py:async_rm). The score is
workloads/math/reward.py:compute_score on the decoded completion and the dataset answer, unchanged. It runs in a
thread so a slow math_verify call doesn't block the rollout event loop. Off the main thread reward.py's SIGALRM
timeouts are off, so the call goes through reward_safe (SHARED_W1 rev. 7): a spawn process pool where the timeouts
work, with a 30 s hard kill that scores 0. Runs before 2026-11-10 10:30 UTC (all W1-S runs) called reward.compute_score
directly.
"""

import asyncio

from reward_safe import compute_score


async def reward(args, sample) -> float:
    return await asyncio.to_thread(compute_score, sample.response, sample.label)
