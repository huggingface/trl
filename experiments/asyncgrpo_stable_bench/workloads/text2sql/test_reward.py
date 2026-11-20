"""CPU check of the ported reward: the gold SQL in a <solution> block must be graded correct (eval score 1.0) on every
sampled task; a wrong query scores 0, no <solution> scores -1, an unsafe query -1. Also prints prompt token lengths.

Usage: python test_reward.py <data dir (convert.py output)> <n tasks per split>
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

import t2sql_episode as ep


def episode(text, unsafe=False):
    return {"turns": [text], "final_text": text, "unsafe": unsafe, "tool_calls": 0, "tool_protocol_events": []}


async def check(task):
    gold = episode(f"<solution>{task['gold_sql']}</solution>")
    out = {
        "gold_eval": await ep.reward_func(task, gold, evaluation=True),
        "gold_train": await ep.reward_func(task, gold, evaluation=False),
        "wrong": await ep.reward_func(task, episode("<solution>SELECT 123456789</solution>"), evaluation=True),
        "none": await ep.reward_func(task, episode("I think the answer is 3."), evaluation=True),
        "unsafe": await ep.reward_func(task, episode("", unsafe=True), evaluation=True),
    }
    return {k: (v["score"], v["correct"], v["error_category"]) for k, v in out.items()}


async def main(data_dir, n):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")
    failures = 0
    for split in ("validation", "train", "hard", "arcwise"):
        tasks = [json.loads(json.loads(line)["task"]) for line in open(data_dir / f"{split}.jsonl")]
        lens = [len(ep.opening_prompt_ids(tok, t)) for t in tasks]
        print(f"{split}: {len(tasks)} tasks, prompt tokens mean {statistics.mean(lens):.0f} max {max(lens)}", flush=True)
        t0 = time.time()
        results = await asyncio.gather(*(check(t) for t in tasks[:n]))
        bad = [(t["id"], r) for t, r in zip(tasks, results) if r["gold_eval"][0] != 1.0 or r["wrong"][0] != 0.0
               or r["none"][0] != -1.0 or r["unsafe"][0] != -1.0]
        failures += len(bad)
        train_scores = [r["gold_train"][0] for r in results]
        print(f"  {len(results)} graded in {time.time() - t0:.0f}s: gold eval score 1.0 on "
              f"{sum(r['gold_eval'][0] == 1.0 for r in results)}/{len(results)}; train-config score of a bare gold "
              f"answer (process penalties apply) mean {statistics.mean(train_scores):.3f}; failures {len(bad)}")
        for task_id, r in bad[:10]:
            print("   FAIL", task_id, r)
    text = tok.decode(ep.opening_prompt_ids(tok, tasks[0]))
    print("---- rendered prompt (arcwise[0]), head and tail ----")
    print(text[:1500])
    print("....")
    print(text[-1200:])
    print("ALL OK" if failures == 0 else f"{failures} FAILURES")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), int(sys.argv[2])))
