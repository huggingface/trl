"""verl W2 run dir -> results/raw/<run_id>/episodes.jsonl and the W2 fields of steps.jsonl (run after collect/verl.py).

Episode records come from two files per episode in <run_dir>/episodes/: `<eid>.jsonl` (written by termgrade_env's
RecordedHarborSession in the Harbor client: phases, turns as Harbor saw them) and `<eid>.verl.json` (written by
configs/verl/w2_agent_loop.py: per-call token counts and exact model time from verl's generate calls).
Usage: python collect/verl_w2.py <run_dir>
"""

import glob
import json
import os
import sys

from agentic import normalize, step_fields


BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    run_dir = os.path.abspath(sys.argv[1])
    out = os.path.join(BENCH, "results", "raw", os.path.basename(run_dir))
    episodes = []
    for path in sorted(glob.glob(os.path.join(run_dir, "episodes", "*.verl.json"))):
        verl = json.load(open(path))
        harbor_path = path[: -len(".verl.json")] + ".jsonl"
        lines = open(harbor_path).read().splitlines() if os.path.exists(harbor_path) else []
        if lines:
            rec = json.loads(lines[-1])
        else:  # the client process died before Harbor answered
            rec = {"start": verl["start"], "end": verl["end"], "reward": None, "error": verl.get("error")}
        e = normalize(rec)
        wall = e["wall_s"]
        e.update(
            {
                "turns": verl["calls"],
                "completion_tokens": verl["completion_tokens"],
                "seq_tokens": verl["seq_tokens"],
                "prompt_tokens_sum": verl["call_tokens_sum"] - verl["completion_tokens"],
                "call_tokens_sum": verl["call_tokens_sum"],
                "loss_tokens": sum(s["loss_tokens"] for s in verl["samples"]),
                "sample_tokens": sum(s["prompt"] + s["response"] for s in verl["samples"]),
                "samples": len(verl["samples"]),
                "overflow": verl["overflow"],
                "model_s": verl["model_s"],
                "sandbox_s": wall - verl["model_s"],
                "slot_wall_s": verl["end"] - verl["start"],
                "eid": verl["eid"],
            }
        )
        episodes.append(e)
    episodes.sort(key=lambda e: e["end"])
    with open(os.path.join(out, "episodes.jsonl"), "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")

    steps_path = os.path.join(out, "steps.jsonl")
    rows = [json.loads(line) for line in open(steps_path)] if os.path.exists(steps_path) else []
    prev = 0.0
    for row in rows:
        t = row.get("time")
        if t is None:
            continue
        window = [e for e in episodes if prev < e["end"] <= t]
        row.update(step_fields(episodes, prev, t))
        forwarded = row.get("trained_tokens")  # verl's perf/total_num_tokens: what it actually forwarded
        shared = sum(e["call_tokens_sum"] for e in window)
        row["forwarded_tokens"] = forwarded
        row["trained_tokens"] = shared  # SHARED_W2 definition: every call's prompt + completion
        row["loss_tokens"] = sum(e["loss_tokens"] for e in window)
        row["prefix_share_ratio"] = shared / forwarded if forwarded else None
        prev = t
    with open(steps_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {out}: {len(episodes)} episodes, {len(rows)} steps")


if __name__ == "__main__":
    main()
