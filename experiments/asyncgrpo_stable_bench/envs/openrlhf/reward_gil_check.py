"""Does the shared W1 reward freeze an asyncio event loop when OpenRLHF runs it via asyncio.to_thread?

OpenRLHF calls `reward_func` with `asyncio.to_thread` inside the vLLM rollout actor (utils/agent.py), so math_verify's
SIGALRM timeouts are off. A ticker coroutine measures the longest gap of the loop while one reward call runs. Each case
runs in its own subprocess with a wall limit so a stuck case can't stall the check.

    python reward_gil_check.py ADAPTER.py [ADAPTER.py ...]   # every case against each reward_func adapter
"""

import asyncio
import importlib.util
import os
import subprocess
import sys
import time


CASES = [  # (boxed answer in the completion, gold)
    ("42", "42"),
    ("x^2+1", "1+x^2"),
    ("9^{9^{9}}", "3"),
    ("10^{10^{10}}", "3"),
    ("2^{2^{2^{2^{2}}}}", "3"),
    ("(10^{9})!", "3"),
    ("1000000!", "3"),
    ("99999999999^{99999999999}", "3"),
]
LIMIT = int(os.environ.get("CASE_LIMIT", "120"))


def load(path):
    spec = importlib.util.spec_from_file_location("reward_func", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reward_func


async def one(path, box, gold):
    reward_func = load(path)
    prompt = "Q: "
    query = prompt + "so the answer is \\boxed{" + box + "}."
    gaps, last, done = [0.0], time.monotonic(), False

    async def ticker():
        nonlocal last
        while not done:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            gaps[0] = max(gaps[0], now - last)
            last = now
            print(f"TICK {now - t0:.1f}", flush=True)

    t0 = time.monotonic()
    t = asyncio.create_task(ticker())
    out = await asyncio.to_thread(reward_func, [query], [prompt], [gold])
    wall = time.monotonic() - t0
    done = True
    await t
    gaps[0] = max(gaps[0], time.monotonic() - last)
    print(f"RESULT reward={out['rewards'].tolist()} wall={wall:.1f}s max_loop_gap={gaps[0]:.1f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        asyncio.run(one(sys.argv[1], sys.argv[2], sys.argv[3]))
        sys.exit(0)
    for path in sys.argv[1:]:
        name = os.path.basename(path)
        for box, gold in CASES:
            t0 = time.monotonic()
            try:
                r = subprocess.run(
                    [sys.executable, __file__, path, box, gold], capture_output=True, text=True, timeout=LIMIT
                )
                res = [l for l in r.stdout.splitlines() if l.startswith("RESULT")] or [
                    f"rc={r.returncode} {r.stderr[-300:]!r}"
                ]
                msg = res[-1]
            except subprocess.TimeoutExpired as e:
                out = (
                    e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                )  # bytes even with text=True
                ticks = [l for l in out.splitlines() if l.startswith("TICK")]
                last_tick = ticks[-1].split()[1] if ticks else "none"
                msg = f"STUCK: no return after {LIMIT}s; event loop last ticked at t={last_tick}s ({len(ticks)} ticks)"
            print(f"{name:20s} box={box!r:32s} gold={gold!r:8s} {msg}  [{time.monotonic() - t0:.0f}s]", flush=True)
