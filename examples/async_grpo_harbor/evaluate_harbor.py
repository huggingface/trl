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

"""Evaluate a policy on held-out Harbor tasks: every task once under every harness, pass@1 per harness.

Rollouts use the training settings of `async_grpo_harbor.py` (sampling policy, step limit, agent timeout), through
the same `HarborSessionFactory`, against a vLLM serving the base model or a checkpoint. Each task/harness pair is
written to `<output-dir>/pairs/` as soon as it finishes; rerunning the same command retries only ungraded pairs.

    python examples/async_grpo_harbor/evaluate_harbor.py \
        --server http://127.0.0.1:8200 --vllm-url http://127.0.0.1:8000 \
        --model Qwen/Qwen3.5-0.8B --split <test-tasks-dir> \
        --harnesses opencode,mini-swe-agent,pi,openhands-sdk --output-dir runs/eval-base

A pair is graded when the task's verifier returned a reward, and solved when that reward is at least 1. Report
`pass_at_1` together with `coverage`: infrastructure failures have no grade and are excluded from `pass_at_1`.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harbor_env.harness import HarborSessionFactory
from openenv.core.harness.capture.upstream import training_sampling


def run_pair(factory, row):
    """Run one rollout and return its grade, tool calls and token counts."""
    start = time.monotonic()
    session = factory.create(row["prompt"])
    try:
        session.wait_for_completion()
        result = {"reward": session.verify([]).env_reward}
        if session.result is None or not session.result.ok:
            result["error"] = session.result.error if session.result is not None else "rollout call failed"
        try:
            trace = session.fetch_training_trace()
        except ValueError:
            pass  # An unusable capture does not change the verifier's grade.
        else:
            entries = trace.to_trace_entries()
            result["model_calls"] = len(trace.turns)
            result["generated_tokens"] = sum(len(turn.completion_token_ids) for turn in trace.turns)
            result["tool_calls"] = sum(
                len(((e.get("response") or {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls") or [])
                for e in entries
            )
    finally:
        session.close()
    result["elapsed_s"] = round(time.monotonic() - start, 1)
    return result


def difficulty(split, task):
    """The task's declared difficulty, recorded in each pair so results stay readable without the task folder."""
    toml = Path(split, "tasks", task, "task.toml")
    m = re.search(r'^difficulty = "(\w+)"', toml.read_text(), re.M) if toml.exists() else None
    return m[1] if m else None


def summarize(rows, expected):
    graded = [row for row in rows if row.get("reward") is not None]

    def mean(key):
        values = [row[key] for row in graded if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "pairs": expected,
        "graded": len(graded),
        "coverage": len(graded) / expected,
        "pass_at_1": sum(row["reward"] >= 1.0 for row in graded) / len(graded) if graded else None,
        "mean_reward": mean("reward"),
        "mean_tool_calls": mean("tool_calls"),
        "mean_model_calls": mean("model_calls"),
        "mean_generated_tokens": mean("generated_tokens"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8200", help="a running `openenv harbor serve` on the test set")
    p.add_argument("--vllm-url", required=True)
    p.add_argument("--model", required=True, help="the name vLLM serves the policy under")
    p.add_argument("--checkpoint", default="", help="label recorded in summary.json, e.g. the checkpoint path")
    p.add_argument("--split", required=True, help="the Harbor task dataset the server was started with")
    p.add_argument("--harnesses", default="opencode,mini-swe-agent,pi,openhands-sdk")
    p.add_argument("--sandbox", default="e2b")
    # Test tasks may report several components (correctness, submission, tool_efficiency) and no `reward`.
    p.add_argument("--reward-key", default="correctness,reward", help="preference order among reward components")
    p.add_argument("--n-tasks", type=int, default=250)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--agent-step-limit", type=int, default=17)
    p.add_argument("--agent-timeout", type=float, default=600.0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    harnesses = [h for h in args.harnesses.split(",") if h]
    output_dir = Path(args.output_dir)
    (output_dir / "pairs").mkdir(parents=True, exist_ok=True)

    # Same factory and sampling policy as training, one per harness.
    sampling = training_sampling({"temperature": args.temperature, "top_p": 1.0, "top_k": -1})
    factories = {
        harness: HarborSessionFactory(
            args.server,
            split=args.split,
            harness=harness,
            sandbox=args.sandbox,
            llm_url=args.vllm_url,
            model=args.model,
            agent_timeout_sec=args.agent_timeout,
            agent_step_limit=args.agent_step_limit,
            reward_key=args.reward_key,
            num_tasks=args.n_tasks,
            sampling=sampling,
        )
        for harness in harnesses
    }
    # Fetch task metadata once per factory, before the threads start creating sessions.
    for factory in factories.values():
        factory.tasks()
    rows = factories[harnesses[0]].prompt_rows()

    def pair_path(row, harness):
        return output_dir / "pairs" / f"{row['task_name']}--{harness}.json"

    pending = []
    for row in rows:
        for harness in harnesses:
            path = pair_path(row, harness)
            if not path.exists() or json.loads(path.read_text()).get("reward") is None:
                pending.append((row, harness))
    print(f"{len(rows)} tasks x {len(harnesses)} harnesses: {len(pending)} pairs to run, {args.concurrency} at a time")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_pair, factories[harness], row): (row, harness) for row, harness in pending}
        for done, future in enumerate(as_completed(futures), 1):
            row, harness = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - record the failure, keep evaluating the other pairs
                result = {"reward": None, "error": f"{type(exc).__name__}: {exc}"}
            record = {"task_name": row["task_name"], "task_index": row["task_index"], "harness": harness,
                      "difficulty": difficulty(args.split, row["task_name"]), **result}
            pair_path(row, harness).write_text(json.dumps(record, indent=2))
            print(f"[{done}/{len(pending)}] {harness:16s} {row['task_name']}  reward={result.get('reward')}", flush=True)

    records = [json.loads(pair_path(row, h).read_text()) for row in rows for h in harnesses if pair_path(row, h).exists()]
    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "sandbox": args.sandbox,
        "temperature": args.temperature,
        "agent_step_limit": args.agent_step_limit,
        "agent_timeout": args.agent_timeout,
        "overall": summarize(records, len(rows) * len(harnesses)),
        "harnesses": {h: summarize([r for r in records if r["harness"] == h], len(rows)) for h in harnesses},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'harness':16s} {'pass@1':>7s} {'coverage':>9s} {'tool calls':>11s}")
    for name, s in [*summary["harnesses"].items(), ("overall", summary["overall"])]:
        pass_at_1 = f"{s['pass_at_1']:.3f}" if s["pass_at_1"] is not None else "n/a"
        tools = f"{s['mean_tool_calls']:.1f}" if s["mean_tool_calls"] is not None else "n/a"
        print(f"{name:16s} {pass_at_1:>7s} {s['coverage']:>9.2f} {tools:>11s}")


if __name__ == "__main__":
    main()
