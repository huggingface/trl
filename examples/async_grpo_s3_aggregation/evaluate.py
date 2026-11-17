# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a served model on information aggregation with the harness in `harness/`; one JSON line per team episode.

    python examples/async_grpo_s3_aggregation/evaluate.py --url http://127.0.0.1:8000 --name base
    python examples/async_grpo_s3_aggregation/analyze_episodes.py runs/base.jsonl

Held-out problems are ids 1,000,000 + i (training draws ids 0..problems-1); the team seed is the episode index, so a
rerun scores the same teams. `--model` is the served model name (a checkpoint served under its own name works too).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from harness import Settings, VLLMSampler, run_team


HELD_OUT = 1_000_000


async def main_async(a: argparse.Namespace) -> None:
    out = Path(a.out or f"runs/{a.name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    sampler = VLLMSampler(a.url, a.model)
    s = Settings(work_tokens=a.work_tokens, final_tokens=a.final_tokens)
    sem = asyncio.Semaphore(a.concurrency)
    done, ok, t0 = 0, 0, time.monotonic()

    async def one(i: int, f) -> None:
        nonlocal done, ok
        async with sem:
            ep = await run_team(sampler, HELD_OUT + i, i, a.n, s)
        f.write(json.dumps({"model": a.model, **ep.summary(with_tokens=a.save_tokens)}) + "\n")
        f.flush()
        done, ok = done + 1, ok + ep.success
        if done % 16 == 0 or done == a.episodes:
            print(f"{done}/{a.episodes} success {ok / done:.1%} ({time.monotonic() - t0:.0f} s)", flush=True)

    with out.open("w") as f:
        await asyncio.gather(*(one(i, f) for i in range(a.episodes)))
    await sampler.aclose()
    p = ok / a.episodes
    print(
        f"{a.name}: N={a.n} success {p:.1%} (SE {math.sqrt(p * (1 - p) / a.episodes):.1%}, {ok}/{a.episodes}) -> {out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=192)
    ap.add_argument("--concurrency", type=int, default=64, help="teams in flight")
    ap.add_argument("--work-tokens", type=int, default=8192)
    ap.add_argument("--final-tokens", type=int, default=1024)
    ap.add_argument("--save-tokens", action="store_true", help="keep each agent's tokens and mask (for transcripts)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
