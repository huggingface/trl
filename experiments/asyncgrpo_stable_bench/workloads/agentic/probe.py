"""Base pass rate of a model on Harbor tasks, through the same OpenEnv server path TRL trains on (terminus-2, HF
Sandboxes, capture proxy, the SHARED_W2 sampling policy). Also the W2 eval: point it at a vLLM serving a checkpoint.

    python probe.py --server http://localhost:PORT --vllm-url http://localhost:VPORT --model Qwen/Qwen3.5-9B \
        --split /path/to/harbor/<set> --samples 4 --concurrency 64 --out probe.jsonl

One JSON line per episode, written by `termgrade_env.RecordedHarborSession` as each finishes (a killed probe keeps what
it has). Sessions run in threads, as TRL's harness loop runs them; nothing CPU-heavy happens on this side.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from termgrade_env import HARNESS, SAMPLING, SANDBOX, TermGradeSessionFactory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--vllm-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--agent-timeout", type=float, default=600.0)
    p.add_argument("--agent-step-limit", type=int, default=30)
    p.add_argument("--n-tasks", type=int, default=0, help="first N tasks of the split; 0 = all")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.environ["W2_EPISODES_FILE"] = args.out

    factory = TermGradeSessionFactory(
        args.server,
        split=args.split,
        harness=HARNESS,
        sandbox=SANDBOX,
        llm_url=args.vllm_url,
        model=args.model,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        sampling=SAMPLING,
        num_tasks=args.n_tasks or None,
    )
    rows = factory.prompt_rows()
    jobs = [row for _ in range(args.samples) for row in rows]  # sample-major: every task once before any twice
    print(
        f"{len(rows)} tasks x {args.samples} samples = {len(jobs)} episodes, {args.concurrency} at a time", flush=True
    )

    def run(row):
        session = factory.create(row["prompt"])
        try:
            session.wait_for_completion()
            return session.result.reward if session.result else None
        finally:
            session.close()

    done, failed, solved, t0 = 0, 0, 0.0, time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        for f in as_completed([pool.submit(run, row) for row in jobs]):
            try:
                reward = f.result()
            except Exception as e:  # noqa: BLE001 - one broken episode must not stop the probe
                print(f"episode crashed: {e!r}", flush=True)
                reward = None
            done += 1
            failed += reward is None
            solved += reward or 0.0
            if done % 16 == 0 or done == len(jobs):
                scored = done - failed
                print(
                    f"[{time.time() - t0:6.0f}s] {done}/{len(jobs)} done, {failed} unscored, "
                    f"mean reward {solved / max(scored, 1):.3f} over scored",
                    flush=True,
                )


if __name__ == "__main__":
    main()
