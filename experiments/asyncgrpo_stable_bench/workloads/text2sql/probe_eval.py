"""vLLM-only probe / eval of miles' text-to-SQL episode (`t2sql_episode.run_episode`) against a running server.

    python probe_eval.py <vllm url> <model> <out dir> <spec> [<spec> ...]

Each spec is `name=<jsonl>:<samples per task>:<temperature>[:<evaluation 0|1>]`. Training-style specs (evaluation 0)
use the training reward (process and tool-format penalties); evaluation specs score the outcome only, as miles'
`--eval-reward-key correct`. Writes <out>/<name>.episodes.jsonl (one line per episode) and <out>/summary.json:
pass rate (`correct`), reward, standard error clustered by task, and the group statistics GRPO cares about (tasks whose
samples are all correct / all wrong, groups with zero reward std).
"""

import asyncio
import json
import math
import os
import sys
import time
from collections import defaultdict

import aiohttp
import numpy as np
from transformers import AutoTokenizer

import t2sql_episode as ep


CONCURRENCY = int(os.environ.get("CONCURRENCY", "512"))


async def main(url, model, out_dir, specs):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    summary_path = os.path.join(out_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 8)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=3600)) as session:
        for spec in specs:
            name, rest = spec.split("=", 1)
            path, n, temperature, *evaluation = rest.split(":")
            n, temperature = int(n), float(temperature)
            evaluation = bool(int(evaluation[0])) if evaluation else temperature == 0
            tasks = [json.loads(json.loads(line)["task"]) for line in open(path)]
            gate = asyncio.Semaphore(CONCURRENCY)
            out = open(os.path.join(out_dir, f"{name}.episodes.jsonl"), "w", buffering=1)

            async def sample(prompt_ids, max_tokens, temperature=temperature):
                payload = ep.completion_payload(model, prompt_ids, max_tokens, temperature)
                for attempt in range(5):
                    try:
                        async with session.post(f"{url}/v1/completions", json=payload) as r:
                            if r.status >= 400:
                                raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
                            return ep.parse_choice(await r.json())
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt == 4:
                            raise
                        print(f"retry {attempt}: {e!r}", flush=True)
                        await asyncio.sleep(5)

            async def one(task):
                async with gate:
                    t0 = time.time()
                    try:
                        result = await ep.run_episode(sample, tokenizer, task, evaluation=evaluation)
                        record = result["record"]
                    except Exception as e:  # recorded, never scored
                        record = {"id": str(task["id"]), "error": repr(e)[:500]}
                    record["wall_s"] = time.time() - t0
                    out.write(json.dumps(record) + "\n")
                    return record

            t0 = time.time()
            records = await asyncio.gather(*(one(t) for t in tasks for _ in range(n)))
            summary[name] = stats(records, n, time.time() - t0, temperature, evaluation, path)
            print(name, json.dumps(summary[name]), flush=True)
            json.dump(summary, open(summary_path, "w"), indent=1)


def stats(records, n, wall, temperature, evaluation, path):
    by_task = defaultdict(list)
    errors = 0
    for r in records:
        if "error" in r:
            errors += 1
            continue
        by_task[r["id"]].append(r)
    correct = np.array([np.mean([r["correct"] for r in rs]) for rs in by_task.values()])
    score = np.array([np.mean([r["score"] for r in rs]) for rs in by_task.values()])
    flat = [r for rs in by_task.values() for r in rs]
    t = len(correct)
    # SE clustered by task: the std of per-task means over sqrt(tasks); with 1 sample per task it is the binomial SE
    se = float(correct.std(ddof=1) / math.sqrt(t)) if t > 1 else float("nan")
    out = {
        "file": path,
        "tasks": t,
        "samples_per_task": n,
        "temperature": temperature,
        "evaluation": evaluation,
        "episodes": len(flat),
        "errors": errors,
        "wall_s": round(wall),
        "pass_rate": float(correct.mean()),
        "pass_rate_se": se,
        "reward_mean": float(score.mean()),
        "turns_mean": float(np.mean([r["turns"] for r in flat])),
        "tool_calls_mean": float(np.mean([r["tool_calls"] for r in flat])),
        "sampled_tokens_mean": float(np.mean([r["sampled_tokens"] for r in flat])),
        "total_tokens_mean": float(np.mean([r["total_tokens"] for r in flat])),
        "total_tokens_p95": float(np.percentile([r["total_tokens"] for r in flat], 95)),
        "no_answer_frac": float(np.mean([r["score"] <= -1 + 1e-9 or r.get("error_category") == "format" for r in flat])),
        "status": {s: sum(r["status"] == s for r in flat) for s in sorted({r["status"] for r in flat})},
        "turn_cut_by_length_frac": float(np.mean([("length" in r["finish_reasons"]) for r in flat])),
    }
    if n > 1:
        rewards = [np.array([r["score"] for r in rs]) for rs in by_task.values()]
        out["tasks_all_correct"] = int((correct == 1).sum())
        out["tasks_all_wrong"] = int((correct == 0).sum())
        out["groups_zero_reward_std"] = int(sum(rw.std() == 0 for rw in rewards))
        out["tasks_pass_in_0.1_0.8"] = int(((correct >= 0.1) & (correct <= 0.8)).sum())
        out["per_task_pass"] = {k: float(np.mean([r["correct"] for r in rs])) for k, rs in by_task.items()}
    return out


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]))
