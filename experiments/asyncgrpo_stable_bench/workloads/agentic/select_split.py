"""Train / eval split from measured base pass rates (probe.jsonl files of one model).

    python select_split.py --probe <run_dir>/probe.jsonl [--probe ...] [--lo 0.2] [--hi 0.8] [--n-eval 32]

A task enters the pool when it has >= 3 scored episodes and its pass rate (mean verifier reward over scored episodes) is
in [lo, hi]. Tasks whose episodes failed unscored more than half the time are left out (harness or sandbox trouble,
not difficulty). `--n-eval` pool tasks (seed 0) become the held-out eval split, the rest the train split. Writes
train_tasks.txt, eval_tasks.txt and split_stats.json next to this file.
"""

import argparse
import collections
import json
import os
import random
import statistics


here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--probe", action="append", required=True)
p.add_argument("--lo", type=float, default=0.2)
p.add_argument("--hi", type=float, default=0.8)
p.add_argument("--n-eval", type=int, default=32)
args = p.parse_args()

by_task = collections.defaultdict(list)
for path in args.probe:
    for line in open(path):
        r = json.loads(line)
        by_task[(r.get("task") or "").split("/")[-1]].append(r)
by_task.pop("", None)
stats = {}
for task, eps in by_task.items():
    scored = [e["reward"] for e in eps if e["reward"] is not None]
    setup = [e["phases"].get("environment_setup") for e in eps if e.get("phases", {}).get("environment_setup")]
    stats[task] = {
        "n": len(eps),
        "scored": len(scored),
        "pass": sum(scored) / len(scored) if scored else None,
        "wall_p50": statistics.median(e["end"] - e["start"] for e in eps),
        "setup_p50": statistics.median(setup) if setup else None,
        "turns_mean": statistics.mean(e["n_turns"] or 0 for e in eps),
        "seq_tokens_mean": statistics.mean(e.get("seq_tokens") or 0 for e in eps),
    }
pool = sorted(
    t for t, s in stats.items() if s["scored"] >= 3 and s["scored"] >= s["n"] / 2 and args.lo <= s["pass"] <= args.hi
)
rng = random.Random(0)
eval_tasks = sorted(rng.sample(pool, min(args.n_eval, len(pool) // 3)))
train_tasks = [t for t in pool if t not in eval_tasks]
open(os.path.join(here, "train_tasks.txt"), "w").writelines(t + "\n" for t in train_tasks)
open(os.path.join(here, "eval_tasks.txt"), "w").writelines(t + "\n" for t in eval_tasks)
passes = [s["pass"] for s in stats.values() if s["pass"] is not None]
summary = {
    "probed_tasks": len(stats),
    "mean_pass": statistics.mean(passes) if passes else None,
    "frac_zero": sum(p == 0 for p in passes) / max(len(passes), 1),
    "frac_one": sum(p == 1 for p in passes) / max(len(passes), 1),
    "in_band": len(pool),
    "train": len(train_tasks),
    "eval": len(eval_tasks),
    "train_mean_pass": statistics.mean(stats[t]["pass"] for t in train_tasks) if train_tasks else None,
    "eval_mean_pass": statistics.mean(stats[t]["pass"] for t in eval_tasks) if eval_tasks else None,
    "per_task": stats,
}
json.dump(summary, open(os.path.join(here, "split_stats.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))
