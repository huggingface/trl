"""Hang-proof wrapper around the shared W1 reward (SHARED_W1 rev. 7). Same interface: `PROMPT`, `compute_score`.

`reward.compute_score` (math_verify) only has working timeouts on a process main thread (SIGALRM). Called from a
worker thread or an event loop, which is how most RL libraries call rewards, an answer like `\\boxed{9^{9^{9}}}`
makes sympy evaluate a huge integer in C while holding the GIL, which freezes the whole calling process (rollout
actor, rollout worker, heartbeat thread). This wrapper runs the unchanged `compute_score` in a small spawn-based
process pool, where it runs on each worker's main thread (timeouts on), and kills any call still running after
HARD_TIMEOUT seconds, scoring it 0. Scores are identical to `reward.compute_score` except for those calls.

Pool workers use the spawn start method, so they re-import the host's `__main__`: the host script must guard its entry
point with `if __name__ == "__main__"` (Ray workers and library launchers do). Tested from a worker thread in
workloads/math/tests/test_reward_safe.py: the three pathological answers return 0.0 in 5-6.5 s with the caller's
loop still ticking, and normal answers score as `reward.compute_score`.
"""

import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool

from reward import PROMPT  # noqa: F401  (re-exported so callers can switch the import only)
from reward import compute_score as _compute_score


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


def compute_score(completion, answer):
    # A daemonic process (e.g. TRL's rollout worker child) can't start a pool. There, call it directly: safe only on
    # the main thread (TRL: make the reward `async def` so it runs on the event loop's main thread, as
    # configs/trl/w1-S.py does).
    if multiprocessing.current_process().daemon:
        return _compute_score(completion, answer)
    for _ in range(2):  # a pool killed for another call's timeout breaks this call too: retry once on a fresh pool
        pool = _get_pool()
        t0 = time.monotonic()
        try:
            score = pool.submit(_compute_score, completion, answer).result(timeout=HARD_TIMEOUT)
            if time.monotonic() - t0 > 3:
                print(
                    f"reward_safe: slow compute_score {time.monotonic() - t0:.1f}s: {completion[-200:]!r}", flush=True
                )
            return score
        except TimeoutError:
            print(f"reward_safe: compute_score over {HARD_TIMEOUT}s, scored 0: {completion[-200:]!r}", flush=True)
            _drop_pool(pool)
            return 0.0
        except BrokenProcessPool:
            _drop_pool(pool)
    return 0.0
