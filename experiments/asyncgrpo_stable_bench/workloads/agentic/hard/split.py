"""Harder W2 split from the 35B probe on the candidate pool (README.md).

    python split.py --probe <run>/probe-ckpt.jsonl --nop <nop-summary.json> --oracle <oracle-summary.json> [--lo 0.1] [--hi 0.65]

Keeps tasks with >= 3 scored episodes (and at least half the episodes scored), pass rate in [lo, hi], not passed by the
nop agent and passed by the oracle (a task missing from a validation summary is kept: validation may still be running).
14 tasks (seed 0) become the held-out split, the rest train. Writes {train,eval}_tasks.txt and split_stats.json here, and
the task dirs into harbor-local-sif/hard-{train,eval} (Apptainer, TRL) and harbor-local/hard-{train,eval} (podman,
prime-rl), copied from the prepared pool with only docker_image changed for the podman copy.
"""

import argparse
import collections
import json
import os
import random
import shutil

here = os.path.dirname(os.path.abspath(__file__))
TG = "/fsx/amine_dirhoussi/stable-bench-src/termgrade"
p = argparse.ArgumentParser()
p.add_argument("--probe", action="append", required=True)
p.add_argument("--nop", default="")
p.add_argument("--oracle", default="")
p.add_argument("--lo", type=float, default=0.1)
p.add_argument("--hi", type=float, default=0.65)
p.add_argument("--n-eval", type=int, default=14)
p.add_argument("--write", action="store_true")
a = p.parse_args()

eps = collections.defaultdict(list)
for path in a.probe:
    for line in open(path):
        r = json.loads(line)
        if r.get("task"):
            eps[r["task"].split("/")[-1]].append(r["reward"])
nop = json.load(open(a.nop)) if a.nop else {}
orc = json.load(open(a.oracle)) if a.oracle else {}
nop_pass = {t for t, v in nop.items() if any(x["reward"] == 1.0 for x in v)}
orc_fail = {t for t, v in orc.items() if not any(x["reward"] == 1.0 for x in v)}
stats = {}
for t, v in eps.items():
    s = [x for x in v if x is not None]
    stats[t] = {"n": len(v), "scored": len(s), "pass": sum(s) / len(s) if s else None}
ok = lambda t: stats[t]["scored"] >= 3 and stats[t]["scored"] >= stats[t]["n"] / 2  # noqa: E731
pool = sorted(t for t in stats if ok(t) and a.lo <= stats[t]["pass"] <= a.hi and t not in nop_pass and t not in orc_fail)
ev = sorted(random.Random(0).sample(pool, min(a.n_eval, len(pool) // 3)))
tr = [t for t in pool if t not in ev]
mean = lambda ts: round(sum(stats[t]["pass"] for t in ts) / max(len(ts), 1), 3)  # noqa: E731
passes = [s["pass"] for t, s in stats.items() if s["pass"] is not None and ok(t)]
hist = collections.Counter(round(x, 2) for x in passes)
summary = {
    "probed": len(stats), "probed_ok": len(passes), "mean_pass_all": round(sum(passes) / max(len(passes), 1), 3),
    "pass_hist": dict(sorted(hist.items())), "nop_pass": sorted(nop_pass & set(stats)), "oracle_fail": sorted(orc_fail & set(stats)),
    "in_band": len(pool), "train": len(tr), "eval": len(ev), "train_mean_pass": mean(tr), "eval_mean_pass": mean(ev),
    "lo": a.lo, "hi": a.hi, "per_task": stats,
}
print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))
if a.write:
    open(f"{here}/train_tasks.txt", "w").writelines(t + "\n" for t in tr)
    open(f"{here}/eval_tasks.txt", "w").writelines(t + "\n" for t in ev)
    json.dump(summary, open(f"{here}/split_stats.json", "w"), indent=1)
    rows = {json.loads(l)["task"]: l for l in open(f"{TG}/harbor-local-sif/hardprobe/tasks.jsonl")}
    for name, ts in (("hard-train", tr), ("hard-eval", ev)):
        for kind, img in (("harbor-local-sif", None), ("harbor-local", "localhost/termgrade-base:latest")):
            out = f"{TG}/{kind}/{name}"
            shutil.rmtree(out, ignore_errors=True)
            for t in ts:
                shutil.copytree(f"{TG}/harbor-local-sif/hardprobe/tasks/{t}", f"{out}/tasks/{t}")
                if img:
                    f = f"{out}/tasks/{t}/task.toml"
                    s = open(f).read()
                    s = "\n".join(f'docker_image = "{img}"' if l.startswith("docker_image") else l for l in s.split("\n"))
                    open(f, "w").write(s)
            open(f"{out}/tasks.jsonl", "w").writelines(rows[t] for t in ts)
        print("wrote", name, len(ts))
