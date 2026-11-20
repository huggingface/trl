"""W2 held-out re-evals at 12 samples per task (local Apptainer boxes) vs the 4-sample HF-Sandbox evals of the same checkpoints.

Writes results/tables/w2_heldout_x12.md and .csv. The header log comes from stable-bench-runs/queue/x12/log.md.
"""

import csv
import glob
import json
import math
import os

RUNS = "/fsx/amine_dirhoussi/stable-bench-runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables", "w2_heldout_x12")
R = RUNS
ROWS = [("base", "9B", 0, "base9b", "Qwen/Qwen3.5-9B"), ("base", "35B-A3B", 0, "base35b", "Qwen/Qwen3.5-35B-A3B")]
for s in (20, 40, 60):
    ROWS += [
        ("TRL (st8)", "9B", s, f"trl9b-{s}", f"{R}/trl/trl-w2-M-local-sharded-st8-372267/checkpoint-{s}"),
        ("prime-rl", "9B", s, f"prl9b-{s}", f"{R}/prime-rl/prime-rl-w2-M-shared-368590/hf/step_{s}"),
        ("verl", "9B", s, f"verl9b-{s}", f"{R}/verl/verl-w2-M-shared-371895/ckpt/global_step_{s}/actor/huggingface"),
        ("TRL (st8)", "35B-A3B", s, f"trl35b-{s}",
         f"{R}/trl-moe-ep/merged/trl-moe-w2-Qwen3.5-35B-A3B-ep4-bfloat16-lora32-local-st8-372251-checkpoint-{s}"),
        ("prime-rl", "35B-A3B", s, f"prl35b-{s}", f"{R}/prime-rl/prime-rl-w2-moe-shared-371663/hf/step_{s}"),
    ]  # fmt: skip
ROWS.sort(key=lambda r: (r[1] != "9B", r[0] != "base", r[0], r[2]))


def finished(d):
    log = os.path.join(d, "probe-ckpt.log")
    if not (os.path.exists(os.path.join(d, "job_end")) and os.path.exists(log)):
        return False
    for line in open(log):
        if "] " in line and " done" in line:
            a, b = line.split("] ", 1)[1].split(" done")[0].split("/")
            if a == b:
                return True
    return False


def stats(d):
    eps = [json.loads(line) for line in open(os.path.join(d, "probe-ckpt.jsonl")) if line.strip()]
    sc = [e for e in eps if e.get("reward") is not None]
    m = sum(e["reward"] for e in sc) / len(sc)
    by_task = {}
    for e in sc:
        by_task.setdefault(e["task"], []).append(e["reward"])
    tm = [sum(v) / len(v) for v in by_task.values()]
    mu = sum(tm) / len(tm)
    se_task = math.sqrt(sum((x - mu) ** 2 for x in tm) / (len(tm) - 1) / len(tm))
    return {"job": d.rsplit("-", 1)[1], "mean": m, "n": len(sc), "total": len(eps), "se": math.sqrt(m * (1 - m) / len(sc)),
            "se_task": se_task, "tasks": len(tm), "budget_stops": sum(1 for e in eps if e.get("budget_stops"))}  # fmt: skip


dirs = []
for d in sorted(glob.glob(f"{RUNS}/trl/trl-w2probe-M-eval-*")):
    env = dict(line.strip().split("=", 1) for line in open(os.path.join(d, "launch.env")) if "=" in line)
    if env.get("SPLIT", "eval") == "eval" and finished(d):
        dirs.append((d, env))
rows = []
for lib, fam, step, key, model in ROWS:
    new = [d for d, env in dirs if env.get("TAG") == f"x12-{key}"]
    # the 4-sample HF-Sandbox eval of the same checkpoint (latest finished one)
    old = [d for d, env in dirs if env.get("EVAL_MODEL") == model and env.get("SAMPLES", "4") == "4" and env.get("SANDBOX", "hf") == "hf"]
    rows.append({"lib": lib, "family": fam, "step": step, "key": key, "model": model,
                 "new": stats(new[-1]) if new else None, "old": stats(old[-1]) if old else None})  # fmt: skip

with open(OUT + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "lib", "step", "model", "eval_job", "mean", "scored", "episodes", "se_binomial", "se_task", "tasks",
                "old_eval_job", "old_mean", "old_scored", "old_se_binomial", "old_se_task"])  # fmt: skip
    for r in rows:
        n, o = r["new"] or {}, r["old"] or {}
        f4 = lambda x: "" if x is None else f"{x:.4f}"  # noqa: E731
        w.writerow([r["family"], r["lib"], r["step"], r["model"], n.get("job", ""), f4(n.get("mean")), n.get("n", ""), n.get("total", ""),
                    f4(n.get("se")), f4(n.get("se_task")), n.get("tasks", ""), o.get("job", ""), f4(o.get("mean")), o.get("n", ""),
                    f4(o.get("se")), f4(o.get("se_task"))])  # fmt: skip


def cell(s):
    return "pending" if s is None else f"{s['mean']:.3f}"


base_new = {r["family"]: r["new"] for r in rows if r["lib"] == "base"}
base_old = {r["family"]: r["old"] for r in rows if r["lib"] == "base"}
md = ["# W2 held-out evals at 12 samples per task", ""]
log = f"{RUNS}/queue/x12/log.md"
if os.path.exists(log):
    md += ["## Log", ""] + [line.rstrip() for line in open(log)] + [""]
md += [
    "TermGrade held-out split (`SPLIT=eval`, 14 tasks), terminus-2, 30 model calls and 600 s per episode, thinking off. Every row ran on "
    "**local Apptainer boxes** (`SANDBOX=local`, `HARBOR_SHARDS=4`, `TERMINUS_MAX_TURNS=1`), 12 episodes per task = 168 per eval. "
    "The post's numbers came from **HF Sandboxes** at 4 episodes per task (56 per eval) without `TERMINUS_MAX_TURNS`. The backend and "
    "the turn cap changed, so the bases were rerun too: these numbers replace the HF-Sandbox ones only as a set, never row by row.",
    "",
    "SE (binomial) = sqrt(p(1-p)/n) over scored episodes. SE (task) = standard error of the mean of the 14 per-task means, which counts "
    "episodes of one task as correlated. Δ base = mean minus the same-size base model in the same set.",
    "",
    "## 12-sample results (local boxes)",
    "",
    "| model | run | step | mean | scored / episodes | SE (binomial) | SE (task) | Δ base |",
    "|---|---|---|---|---|---|---|---|",
]  # fmt: skip
for r in rows:
    n = r["new"]
    if n is None:
        md.append(f"| {r['family']} | {r['lib']} | {r['step'] or '-'} | pending | | | | |")
        continue
    b = base_new.get(r["family"])
    d = "" if r["lib"] == "base" or not b else f"{n['mean'] - b['mean']:+.3f}"
    md.append(f"| {r['family']} | {r['lib']} | {r['step'] or '-'} | {n['mean']:.3f} | {n['n']} / {n['total']} | {n['se']:.3f} | {n['se_task']:.3f} | {d} |")
md += ["", "## Against the post's 4-sample HF-Sandbox numbers", "",
       "| model | run | step | old mean (56 ep.) | old SE (bin / task) | new mean (168 ep.) | new SE (bin / task) | new - old | old Δ base | new Δ base |",
       "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
for r in rows:
    n, o = r["new"], r["old"]
    diff = "" if not (n and o) else f"{n['mean'] - o['mean']:+.3f}"
    ob, nb = base_old.get(r["family"]), base_new.get(r["family"])
    odb = "" if r["lib"] == "base" or not (o and ob) else f"{o['mean'] - ob['mean']:+.3f}"
    ndb = "" if r["lib"] == "base" or not (n and nb) else f"{n['mean'] - nb['mean']:+.3f}"
    ose = "" if not o else f"{o['se']:.3f} / {o['se_task']:.3f}"
    nse = "" if not n else f"{n['se']:.3f} / {n['se_task']:.3f}"
    md.append(f"| {r['family']} | {r['lib']} | {r['step'] or '-'} | {cell(o)} | {ose} | {cell(n)} | {nse} | {diff} | {odb} | {ndb} |")
md += ["", "Eval jobs (new / old): " + ", ".join(f"{r['key']} {(r['new'] or {}).get('job', '-')} / {(r['old'] or {}).get('job', '-')}" for r in rows), ""]
open(OUT + ".md", "w").write("\n".join(md))
print("\n".join(md))
