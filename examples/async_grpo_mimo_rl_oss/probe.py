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
# /// script
# dependencies = [
#     "mimoagent @ git+https://github.com/XiaomiMiMo/mimoagent.git",
#     "huggingface_hub>=1.31",
#     "pandas",
#     "pyarrow",
# ]
# ///

"""Probe a policy on `XiaomiMiMo/MiMo-V2.6-RL-oss`, no training.

Three domains, all driven by MiMo's own agent (`mimoagent`, Bash/Read/Write/Edit/Grep/Glob) in one Hugging Face
sandbox per rollout, with the prompts, tool config and step limits of the matching `config/agent/*.yaml` in
`XiaomiMiMo/verl`:

- `general` — 925 knowledge-work tasks (accounting, legal, HR, healthcare ops), business systems served as MCP
  tools, graded by a rubric an LLM judges against a hidden answer key,
- `code` — 2698 software-engineering tasks, graded by applying a hidden test patch and running the task's tests,
- `cyber` — 1000 ARVO crash reproductions, graded by whether the submitted input crashes the named function.

Prints the reward distribution, the pass rate, the mean per task and how many tasks have a reward spread across
their samples, which is the signal GRPO trains on. Writes one JSON line per rollout and one transcript per rollout.

```sh
vllm serve XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --max-model-len 40960 \
    --enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo --generation-config vllm

HF_TOKEN=... python examples/async_grpo_mimo_rl_oss/probe.py --domain code \
    --model XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --n-tasks 32 --samples-per-task 4 --output probe.jsonl
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import general_domain
import swe_domain


def run_one(factory, task: dict, sample: int) -> dict:
    t0 = time.perf_counter()
    result = {
        "instance_id": task["instance_id"],
        "sample": sample,
        "category": task["category"],
        "language": task["language"],
        "tier": task["tier"],
        "exit_status": None,
        "steps": 0,
        "tool_calls": {},
        "format_errors": 0,
        "blocked": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "calls": [],
        "final_answer": "",
        "reward": None,
        "reward_error": None,
        "judge_error": None,
        "items": {},
    }
    session = None
    try:
        session = factory.create(task, sample)
        exit_status, exit_message = session.run()
        agent = session.agent
        calls = Counter(
            call["function"]["name"]
            for message in agent.messages
            if message["role"] == "assistant"
            for call in message.get("tool_calls") or []
        )
        result.update(
            exit_status=exit_status,
            steps=agent._steps_taken,
            tool_calls=dict(calls),
            format_errors=sum(agent.tool_call_errors),
            blocked=len(agent.antihack.blocks),
            input_tokens=agent.model.token_stats.input_tokens,
            output_tokens=agent.model.token_stats.output_tokens,
            calls=agent.model.calls,
            final_answer=exit_message,
        )
        result.update(session.grade(exit_message))
    except (
        Exception
    ) as e:  # a sandbox that failed to start or died mid-rollout is a failed rollout, not a failed probe
        result.update(exit_status=type(e).__name__, error=str(e)[:500])
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
    result["wall_s"] = time.perf_counter() - t0
    print(
        f"[rollout] {result['instance_id']}#{sample} {result['exit_status']} steps={result['steps']} "
        f"reward={result['reward']} {result['reward_error'] or ''} {result['wall_s']:.0f}s",
        flush=True,
    )
    return result


def summarize(results: list[dict], tasks: list[dict]) -> None:
    scored = [r for r in results if r["reward"] is not None]
    masked = Counter(r["reward_error"] or r["exit_status"] for r in results if r["reward"] is None)
    print(f"\nrollouts: {len(results)}, scored: {len(scored)}, unscored: {dict(masked)}")
    print("exit statuses:", dict(Counter(r["exit_status"] for r in results)))
    if results:
        print(
            f"mean steps: {statistics.mean(r['steps'] for r in results):.1f}, "
            f"mean output tokens: {statistics.mean(r['output_tokens'] for r in results):.0f}, "
            f"mean wall: {statistics.mean(r['wall_s'] for r in results):.0f}s"
        )
        print("tool calls:", dict(sum((Counter(r["tool_calls"]) for r in results), Counter()).most_common(12)))
    summarize_packing(results)
    if not scored:
        return
    # Upstream trains on the binarized reward (`REWARD_BINARIZE_THRESHOLD=1.0`): a rollout passes only when every
    # rubric item does, so the pass rate is the number to read, the weighted score is the shape below it.
    rewards = [r["reward"] for r in scored]
    passed = [x >= 1 - 1e-6 for x in rewards]
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    bins = Counter(sum(x > edge for edge in edges) for x in rewards)
    labels = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1)", "1"]
    hist = ", ".join(f"{labels[k]}: {bins[k]}" for k in range(len(labels)) if bins[k])
    print(
        f"pass rate: {sum(passed)}/{len(scored)} = {statistics.mean(passed):.3f}; weighted score mean {statistics.mean(rewards):.3f}, histogram {hist}"
    )

    # Only General grades per rubric item; the other domains return one verdict and no `verifier_meta.json`.
    if all("task_dir" in task for task in tasks):
        by_method: dict[str, list[float]] = defaultdict(list)
        methods = {}
        for task in tasks:
            for item in json.loads((task["task_dir"] / "verifier_meta.json").read_text())["items"]:
                methods[(task["instance_id"], item["id"])] = item.get("method", "?")
        for r in scored:
            for item_id, score in r["items"].items():
                by_method[methods.get((r["instance_id"], item_id), "gate")].append(score)
        print(
            "rubric items by method:",
            {m: f"{statistics.mean(v):.2f} ({len(v)})" for m, v in sorted(by_method.items())},
        )

    by_category: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_category[r["category"]].append(r["reward"])
    print("by category:", {c: f"{statistics.mean(v):.2f} ({len(v)})" for c, v in sorted(by_category.items())})

    by_task: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_task[r["instance_id"]].append(r["reward"])
    for instance_id, values in sorted(by_task.items()):
        n_pass = sum(x >= 1 - 1e-6 for x in values)
        print(
            f"  pass {n_pass}/{len(values)}  score {statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}  {instance_id}"
        )
    mixed = sum(0 < sum(x >= 1 - 1e-6 for x in v) < len(v) for v in by_task.values())
    spread = sum(max(v) - min(v) > 1e-6 for v in by_task.values())
    print(
        f"tasks with both a pass and a fail (GRPO signal on the binarized reward): {mixed}/{len(by_task)}; with any score spread: {spread}/{len(by_task)}"
    )


def summarize_packing(results: list[dict]) -> None:
    """What tree packing would save on these rollouts. Every turn re-sends the conversation, so sequence packing
    forwards the sum of all calls while a rollout folded into a prefix forest forwards about its final context; the
    samples of a task then share the opening prompt (system, tools, task) on top of that."""
    rollouts = [r for r in results if r["calls"]]
    if not rollouts:
        return
    forwarded = sum(sum(i + o for i, o in r["calls"]) for r in rollouts)
    context = sum(r["calls"][-1][0] + r["calls"][-1][1] for r in rollouts)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        by_task[r["instance_id"]].append(r)
    group_unique = sum(
        sum(r["calls"][-1][0] + r["calls"][-1][1] for r in group) - (len(group) - 1) * group[0]["calls"][0][0]
        for group in by_task.values()
    )
    print(
        f"packing: mean turns {statistics.mean(len(r['calls']) for r in rollouts):.1f}, "
        f"mean forwarded tokens {forwarded / len(rollouts):.0f} vs mean final context {context / len(rollouts):.0f}; "
        f"ratio within a rollout {forwarded / context:.2f}x, with the task's samples in one row {forwarded / group_unique:.2f}x"
    )


def make_factory(domain: str, args, transcripts_dir: Path):
    domain_timeout = 1200 if domain == "general" else swe_domain.DOMAINS[domain]["agent_timeout"]
    common = dict(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        chat_template_kwargs={} if args.enable_thinking else {"enable_thinking": False},
        temperature=args.temperature,
        top_p=args.top_p,
        max_turn_tokens=args.max_turn_tokens,
        max_observation_length=args.max_observation_length,
        agent_timeout=args.agent_timeout or domain_timeout,
        verify_timeout=args.verify_timeout,
        flavor=args.sandbox_flavor,
        transcripts_dir=transcripts_dir,
    )
    if domain == "general":
        return general_domain.GeneralTaskSessionFactory(
            step_limit=args.step_limit,
            judge_env=general_domain.judge_env(args.judge_url, args.judge_key, args.judge_model),
            **common,
        )
    return swe_domain.SweTaskSessionFactory(domain=domain, **common)


def load_tasks(domain: str, args) -> list[dict]:
    if domain == "general":
        return general_domain.load_tasks(
            args.n_tasks,
            args.seed,
            set(args.categories.split(",")) if args.categories else None,
            None if args.language == "all" else args.language,
        )
    return swe_domain.load_tasks(domain, args.n_tasks, args.seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", default="general", choices=["general", "code", "cyber"])
    p.add_argument("--model", default="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="trl")
    p.add_argument("--output", default="probe.jsonl")
    p.add_argument("--n-tasks", type=int, default=32)
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--categories", default=None)  # general only, e.g. accounting_audit_tax,legal_compliance
    p.add_argument("--language", default="en", choices=["en", "zh", "all"])  # general only
    p.add_argument("--max-inflight", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)  # upstream general.sh: 1.0, top_p 0.95
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-turn-tokens", type=int, default=4096)
    # Characters kept per tool result. mimoagent's 40000 is sized for MiMo's 262k context; at a 40960-token window
    # two parallel reads of a CSV would fill the whole window in one turn.
    p.add_argument("--max-observation-length", type=int, default=8000)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--step-limit", type=int, default=500)  # general only; the others carry their own
    # `TRAJECTORY_TIMEOUT` upstream, which differs per domain: 1200 s for General, 4800 for Code and 28800 for
    # Cyber. Left unset, each domain takes its own; a value here overrides all of them.
    p.add_argument("--agent-timeout", type=int, default=None)
    p.add_argument("--verify-timeout", type=int, default=900)
    p.add_argument("--sandbox-flavor", default="cpu-basic")
    p.add_argument("--judge-url", default="https://router.huggingface.co/v1")
    p.add_argument("--judge-model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    p.add_argument("--judge-key", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    logging.getLogger("mimoagent").setLevel(logging.WARNING)
    transcripts_dir = Path(args.output).with_suffix(".transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.domain, args)
    print(f"{len(tasks)} {args.domain} tasks: {dict(Counter(t['category'] for t in tasks))}", flush=True)
    factory = make_factory(args.domain, args, transcripts_dir)

    jobs = [(task, sample) for task in tasks for sample in range(args.samples_per_task)]
    results = []
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool, open(args.output, "w") as out:
        for future in as_completed([pool.submit(run_one, factory, *job) for job in jobs]):
            result = future.result()
            results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
    summarize(results, tasks)


if __name__ == "__main__":
    main()
