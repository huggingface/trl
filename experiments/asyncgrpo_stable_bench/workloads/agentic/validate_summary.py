"""Summarize a `harbor run` jobs dir (oracle or nop validation): reward per task, failures, phase times.

python validate_summary.py <jobs_dir> [--out summary.json]
"""

import argparse
import collections
import json
import os
import statistics
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "collect"))
from agentic import episodes_from_trials  # noqa: E402


p = argparse.ArgumentParser()
p.add_argument("jobs_dir")
p.add_argument("--out", default="")
args = p.parse_args()
recs = []
for root, dirs, files in os.walk(args.jobs_dir):
    if "config.json" in files and any(os.path.exists(os.path.join(root, d, "result.json")) for d in dirs):
        recs += episodes_from_trials(root)
by_task = collections.defaultdict(list)
for r in recs:
    by_task[(r["task"] or "").split("/")[-1]].append(r)
print(f"{len(recs)} trials over {len(by_task)} tasks")
rewards = collections.Counter(str(r["reward"]) for r in recs)
print("rewards", dict(rewards))
errs = collections.Counter((r["exception_type"] or "") for r in recs if r["exception_type"])
print("exceptions", dict(errs))
for key in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
    v = [r["phases"][key] for r in recs if key in r["phases"]]
    if v:
        print(
            f"phase {key}: n {len(v)} p50 {statistics.median(v):.1f} p90 {sorted(v)[int(0.9 * (len(v) - 1))]:.1f} max {max(v):.1f}"
        )
walls = [r["end"] - r["start"] for r in recs]
if walls:
    print(f"trial wall p50 {statistics.median(walls):.1f} max {max(walls):.1f}")
bad = {
    t: [(x["reward"], x["exception_type"], (x["error"] or "")[:200]) for x in v]
    for t, v in by_task.items()
    if any(x["reward"] != 1.0 for x in v)
}
print("tasks not at reward 1:", len(bad))
for t, v in sorted(bad.items()):
    print(" ", t, v)
if args.out:
    json.dump(
        {
            t: [{"reward": x["reward"], "exception": x["exception_type"], "phases": x["phases"]} for x in v]
            for t, v in by_task.items()
        },
        open(args.out, "w"),
        indent=1,
    )
