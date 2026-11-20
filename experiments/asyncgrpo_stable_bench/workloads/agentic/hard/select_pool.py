"""Candidate pool for a harder W2 split for Qwen3.5-35B-A3B (README.md).

TermGrade `unseen` tasks with a fast setup.sh (workloads/agentic/fast_pool.txt), minus the current W2 train/eval split
and the nop-pass tasks, minus tasks the 9B already solved >= 0.5 in probe 364979 (the 35B scores ~0.76 on 9B-mid tasks),
and keeping Qwen3.6-27B's published terminus-2 pass rate in [1/8, 6/8] plus 40 random tasks it never solved.
"""

import collections
import csv
import json
import random
import sys

A = "/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/agentic"
fast = [l.strip() for l in open(f"{A}/fast_pool.txt") if l.strip()]
drop = {l.strip() for f in ("train_tasks.txt", "eval_tasks.txt", "nop_pass_tasks.txt") for l in open(f"{A}/{f}")}
summ = {r["task"]: r for r in csv.DictReader(open("/fsx/amine_dirhoussi/stable-bench-src/termgrade/task_summary.csv"))}
nine = collections.defaultdict(list)
for line in open("/fsx/amine_dirhoussi/stable-bench-runs/trl/trl-w2probe-M-probe-364979/probe-M.jsonl"):
    r = json.loads(line)
    if r.get("task") and r["reward"] is not None:
        nine[r["task"]].append(r["reward"])
mid, zero = [], []
for t in fast:
    if t in drop or summ[t]["oracle_verified"] != "True":
        continue
    if nine.get(t) and sum(nine[t]) / len(nine[t]) >= 0.5:
        continue
    q = float(summ[t]["qwen"])
    if 0.1 <= q <= 0.76:
        mid.append(t)
    elif q == 0.0:
        zero.append(t)
pick = sorted(mid + random.Random(0).sample(zero, min(40, len(zero))))
open(sys.argv[1], "w").writelines(t + "\n" for t in pick)
print(f"mid {len(mid)} zero {len(zero)} -> {len(pick)} candidates")
