"""Summarize probe / eval episodes: pass rate, failures by type, episode shape, and where the time goes.

python analyze_probe.py <run_dir>/probe-M.jsonl [<run_dir>/probe-S.jsonl ...]   (hopper-cpu)
"""

import collections
import os
import statistics
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "collect"))
from agentic import normalize, read_episode_records  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def fmt(name, xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return f"{name}: n/a"
    return f"{name}: mean {statistics.mean(xs):.1f} p50 {pct(xs, 0.5):.1f} p90 {pct(xs, 0.9):.1f} max {max(xs):.1f}"


for path in sys.argv[1:]:
    recs = read_episode_records(path)
    eps = [normalize(r) for r in recs]
    print(f"===== {path}: {len(eps)} episodes")
    status = collections.Counter(e["status"] for e in eps)
    print("status", dict(status))
    scored = [e for e in eps if e["reward"] is not None]
    if scored:
        print(
            f"pass@1 over scored {statistics.mean(e['reward'] for e in scored):.3f}; counting failed as 0: "
            f"{sum(e['reward'] for e in scored) / len(eps):.3f}"
        )
    errs = collections.Counter(
        ((r.get("exception_type") or "") + ": " + (r.get("error") or "")[:90]) for r in recs if r.get("reward") is None
    )
    for k, v in errs.most_common(8):
        print(f"  unscored x{v}: {k}")
    budget = collections.Counter((r.get("exception_type") or "none") for r in recs if r.get("reward") is not None)
    print("exception types among scored", dict(budget))
    by_task = collections.defaultdict(list)
    for e in scored:
        by_task[e["task"]].append(e["reward"])
    rates = [statistics.mean(v) for v in by_task.values() if len(v) >= 3]
    if rates:
        bins = collections.Counter(round(r, 2) for r in rates)
        print(f"tasks with >=3 scored: {len(rates)}; per-task pass-rate histogram {dict(sorted(bins.items()))}")
        print(
            f"  frac 0: {sum(r == 0 for r in rates) / len(rates):.2f}, frac 1: {sum(r == 1 for r in rates) / len(rates):.2f}, "
            f"in [0.2, 0.8]: {sum(0.2 <= r <= 0.8 for r in rates)}"
        )
    print(fmt("wall_s", [e["wall_s"] for e in eps]))
    print(fmt("turns", [e["turns"] for e in scored]))
    print(fmt("seq_tokens (last call prompt+completion)", [e["seq_tokens"] for e in scored]))
    print(fmt("completion_tokens (sum over turns)", [e["completion_tokens"] for e in scored]))
    print(fmt("prompt_tokens_sum (sum over turns)", [e["prompt_tokens_sum"] for e in scored]))
    for key in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
        print(fmt(f"phase {key}", [e["phases"].get(key) for e in eps]))
    print(fmt("model_s", [e["model_s"] for e in scored]))
    print(fmt("sandbox_s", [e["sandbox_s"] for e in scored]))
    finish = collections.Counter(f for r in recs for f in (r.get("finish_reasons") or []))
    print("finish reasons", dict(finish))
    by_reward = collections.defaultdict(list)
    for e in scored:
        by_reward[e["reward"]].append(e)
    for rw, es in sorted(by_reward.items()):
        print(
            f"  reward {rw}: n {len(es)}, turns mean {statistics.mean(e['turns'] or 0 for e in es):.1f}, wall mean {statistics.mean(e['wall_s'] for e in es):.0f}s"
        )
