"""Line up TRL's weight-sync pause against vLLM's PAUSED_ALL/UNPAUSED lines and the generate spans.

Usage: python timeline.py <trl run dir> [<prime-rl run dir>]
"""

import glob
import json
import re
import sys
from datetime import UTC, datetime

import numpy as np


run = sys.argv[1]
TS = re.compile(
    r"\(EngineCore_DP(\d+) pid=\d+\) INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[scheduler.py:\d+\] setting pause state to (\w+)"
)


def parse_ts(s):
    return datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()


def engine_events(path):
    ev = []  # (t, dp, state)
    with open(path, errors="replace") as f:
        for line in f:
            m = TS.search(line)
            if m:
                ev.append((parse_ts(m.group(2)), int(m.group(1)), m.group(3)))
    return ev


ev = engine_events(f"{run}/vllm.log")
paused = sorted(t for t, _, s in ev if s == "PAUSED_ALL")
unpaused = sorted(t for t, _, s in ev if s == "UNPAUSED")

spans = [json.loads(l) for l in open(f"{run}/spans-train.jsonl")]
syncs = [r for r in spans if r["kind"] == "weight_sync"]
train_s = {r["step"]: r["end"] - r["start"] for r in spans if r["kind"] == "train_step"}
metrics = {}
for l in open(f"{run}/metrics.jsonl"):
    r = json.loads(l)
    if "perf/weight_sync_pause_s" in r:
        metrics[r["step"]] = r
gen = []
for p in glob.glob(f"{run}/spans-generate-*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if r.get("kind") == "generate":
            gen.append((r["start"], r["end"], r["tokens"]))
gen = np.array(gen)
starts, ends, toks = gen[:, 0], gen[:, 1], gen[:, 2]
print(f"{run}: {len(syncs)} syncs, {len(gen)} generate spans, {len(paused) // 4} PAUSED_ALL x4")

rows = []
for s in syncs:
    step = s["step"]
    if step not in metrics:
        continue
    t0 = s["start"]
    pause_s = metrics[step]["perf/weight_sync_pause_s"]
    t_pause = t0 + pause_s
    # first PAUSED_ALL line at or after t0 (1 s log resolution)
    pa = [t for t in paused if t >= int(t0)]
    un = [t for t in unpaused if t >= int(t0)]
    if not pa:
        continue
    t_pa, t_un = pa[0], un[0]
    inflight = int(((starts <= t0) & (ends > t0)).sum())
    # replies the client received during the pause call
    win = (ends > t0) & (ends <= t_pause)
    # replies received >= 2 s after every engine had stopped stepping: they finished on the GPU before the pause and
    # waited in the API server's output path
    late = (ends >= t_pa + 2) & (ends <= t_un + 1)
    prev = (ends > t0 - 60) & (ends <= t0)
    rows.append(
        dict(
            step=step,
            pause_s=pause_s,
            to_paused_all=t_pa - t0,
            paused_all_to_return=t_pause - t_pa,
            paused_all_to_unpaused=t_un - t_pa,
            inflight=inflight,
            replies_in_pause=int(win.sum()),
            late_replies=int(late.sum()),
            late_max_s=float((ends[late] - t_pa).max()) if late.any() else 0.0,
            late_tokens=int(toks[late].sum()),
            replies_per_s_prev60=prev.sum() / 60,
            tok_per_s_prev60=toks[prev].sum() / 60,
            mean_len_prev60=float(toks[prev].mean()) if prev.any() else 0.0,
            long_replies_prev60=int((toks[prev] >= 8192).sum()),
            train_step_s=train_s.get(step, float("nan")),
            queue_size=float(metrics[step].get("sample/rollout_queue_size", float("nan"))),
            concurrency=float(metrics[step].get("rollout/concurrency", float("nan"))),
            vllm_running=float(metrics[step].get("rollout/vllm_running_requests", float("nan"))),
            kv_usage=float(metrics[step].get("rollout/kv_cache_usage", float("nan"))),
            gen_tok_s=float(metrics[step].get("rollout/generated_tok_s", float("nan"))),
        )
    )

keys = list(rows[0])
print("\t".join(keys))
for r in rows:
    print("\t".join(f"{r[k]:.1f}" if isinstance(r[k], float) else str(r[k]) for k in keys))


def rank(x):
    return np.argsort(np.argsort(x))


p = np.array([r["pause_s"] for r in rows])
print(f"\npause_s mean {p.mean():.1f} median {np.median(p):.1f} max {p.max():.1f}; n={len(p)}")
for k in ("to_paused_all", "paused_all_to_return"):
    v = np.array([r[k] for r in rows])
    print(f"{k}: mean {v.mean():.1f} median {np.median(v):.1f} share of pause {v.sum() / p.sum():.0%}")
print("Spearman rho vs pause_s:")
for k in keys[5:]:
    v = np.array([r[k] for r in rows], dtype=float)
    if np.isnan(v).any():
        continue
    print(f"  {k:24s} {np.corrcoef(rank(p), rank(v))[0, 1]:+.2f}")

if len(sys.argv) > 2:
    prl = sys.argv[2]
    pev = engine_events(f"{prl}/logs/attempt_1/inference.log")
    pa = sorted(t for t, _, s in pev if s == "PAUSED_ALL")[::4]
    un = sorted(t for t, _, s in pev if s == "UNPAUSED")[::4]
    gaps = np.array([u - a for a, u in zip(pa, un, strict=False)])
    vals = []
    last = None
    for l in open(f"{prl}/monitors/file/metrics.jsonl"):
        m = re.search(r'"watcher/pause/max": ([0-9.e-]+)', l)
        if m and m.group(1) != last:
            last = m.group(1)
            vals.append(float(last))
    vals = np.array(vals)
    print(
        f"\nprime-rl {prl}: {len(pa)} pauses; PAUSED_ALL->UNPAUSED (1 s res) mean {gaps.mean():.2f} max {gaps.max():.0f}"
    )
    print(
        f"prime-rl /pause per-engine max (distinct values) n={len(vals)} mean {vals.mean():.3f} median {np.median(vals):.3f} p90 {np.quantile(vals, 0.9):.3f} max {vals.max():.3f}"
    )
