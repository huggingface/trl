"""OpenRLHF `--reward.remote_url` adapter for the shared W1 reward (workloads/math/reward.py, called unchanged).

OpenRLHF loads this file and calls `reward_func(queries, prompts, labels)` once per rollout, in a worker thread
of the vLLM rollout actor. `queries` is prompt + completion decoded with special tokens, `prompts` the chat-templated
prompt string, `labels` the dataset `answer` column.

`compute_score` runs in a small process pool, not in OpenRLHF's thread: there it runs on a process main thread, so
math_verify's 5 s SIGALRM timeouts are on (the same as the reference on its main thread), and a call that ignores
SIGALRM anyway (a big-int power evaluated in C holds the GIL) is killed after HARD_TIMEOUT s and scored 0. Run in
the actor's thread, such a call froze the whole rollout actor's event loop and hung the run
(envs/openrlhf/NOTES.md item 19).
"""

import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool

import torch


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "workloads", "math"))
from reward import compute_score  # noqa: E402


HARD_TIMEOUT = 30
_pool, _lock = None, threading.Lock()


def _get_pool():
    global _pool
    with _lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn"))
        return _pool


def _drop_pool(pool):
    global _pool
    with _lock:
        if _pool is pool:
            for p in list(pool._processes.values()):
                p.kill()
            pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


def _score(completion, label):
    for _ in range(2):  # a pool killed for another call's timeout breaks this call too: retry once on a fresh pool
        pool = _get_pool()
        t0 = time.monotonic()
        try:
            score = pool.submit(compute_score, completion, label).result(timeout=HARD_TIMEOUT)
            if time.monotonic() - t0 > 3:  # evidence for item 19: answers that would have stalled the actor's thread
                print(f"w1_reward: slow compute_score {time.monotonic() - t0:.1f}s: {completion[-200:]!r}", flush=True)
            return score
        except TimeoutError:
            print(f"w1_reward: compute_score over {HARD_TIMEOUT}s, scored 0: {completion[-200:]!r}", flush=True)
            _drop_pool(pool)
            return 0.0
        except BrokenProcessPool:
            _drop_pool(pool)
    return 0.0


def reward_func(queries, prompts, labels, **kwargs):
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels, strict=False):
        completion = query[len(prompt) :] if query.startswith(prompt) else query
        rewards.append(_score(completion, label))
    rewards = torch.tensor(rewards, dtype=torch.float)
    return {"rewards": rewards, "scores": rewards, "extra_logs": {}}
