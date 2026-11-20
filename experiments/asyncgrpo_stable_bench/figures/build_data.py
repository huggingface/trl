"""Everything the blog figures plot, gathered into figures/data/bench.json (static charts and the interactive page read it).

    python figures/build_data.py      # in a hopper-cpu job, after collect/throughput_fair.py

Sources: results/raw/<run_id>/ (collect/SCHEMA.md), results/tables/throughput_fair.csv, results/robustness/failures.jsonl,
and, for runs that aren't collected yet, their native files under stable-bench-runs/ (W2 held-out probe logs, the MoE run,
TRL's metrics.jsonl while a run is live). Extends stable-bench-runs/status/dash/gather.py.

Only runs on each library's built-in options count (SHARED_W1 rev. 8): a run whose meta.json carries a `validity` note
(patched library, custom code in the algorithm path) is listed under `excluded` and never plotted. Recommended arms are
not featured. A run or checkpoint that is expected but has no file yet is reported as in progress, so the charts fill in
on the next build without any edit here.
"""

import csv
import glob
import json
import math
import os
import re
import time


BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BENCH, "results", "raw")
RUNS = os.environ.get("RUNS", "/fsx/amine_dirhoussi/stable-bench-runs")
OUT = os.path.join(BENCH, "figures", "data", "bench.json")
SKIP_W1 = 5  # W1 steady state: steps after 5
SKIP_W2 = 1  # W2 steady state: steps after 1
W1_STEPS = 200
W1_CKPTS = [50, 100, 150, 200]

LIBS = ["trl", "prime-rl", "verl", "openrlhf", "skyrl", "miles"]
LIB_NAME = {"trl": "TRL", "prime-rl": "prime-rl", "verl": "verl", "openrlhf": "OpenRLHF", "skyrl": "SkyRL", "miles": "miles"}
# Libraries expected in each W1 size, and what to say where a library has no built-in run.
W1_EXPECTED = {
    "S": ["trl", "prime-rl", "verl", "openrlhf", "skyrl"],
    "M": ["trl", "prime-rl", "openrlhf", "skyrl"],
}
W1_NOT_RUN = {
    "S": {"miles": "not run as shipped: its only correct attention path on B200 is ~7x slower (smoke only)"},
    "M": {"verl": "not run at 9B", "miles": "not run at 9B"},
}
COLLAPSED = {"skyrl"}  # W1 runs that collapsed (built-in rollout_is loss, exit code 0): drawn as they are, labelled
W2_LABEL = {  # run id prefix -> (short label, long label)
    "trl-w2-M-local-sharded-": ("TRL", "TRL, local Apptainer containers, Harbor server sharded, call cap passed through a launcher wrapper"),
    "trl-w2-M-local-": ("TRL before the Harbor fix", "TRL, local Apptainer containers, one Harbor server for all 64 agents"),
    "prime-rl-w2-M-shared-": ("prime-rl", "prime-rl, local podman containers"),
}  # fmt: skip


W2_NOT_RUN = {  # libraries with no W2 run on built-in options, and why (shown as a labelled row, never a silent gap)
    ("skyrl", "9B"): "not runnable as shipped",
}
W2_NOT_RUN_DETAIL = {
    ("skyrl", "9B"): "SkyRL's stock Harbor generator zeroes a whole group when one episode hits the agent time limit (~40-50% of 9B "
                     "episodes), and its built-in fully-async loss collapsed on W1.",
}
FAIL_SHORT = {"oom": "OOM at the shared 65k context", "sandbox": "sandboxes failed to start"}


W2_MAIN = {  # the run that stands for a library in the main W2 charts, when several exist (user decision 2026-11-11)
    ("trl", "9B"): r"trl-w2-M-local-sharded-st8-\d+",  # max_staleness 8, matching prime-rl (user decision)
    ("trl", "35B-A3B"): r"trl-moe-w2-.*-local-st8-\d+",
}


def pick_main(cands):
    """One main run per (library, model). An explicit W2_MAIN pattern wins; otherwise the best-documented run:
    collected and completed, then collected, then a live dir with steps, latest job first."""
    out = {}
    for c in cands:
        out.setdefault((c["lib"], c["family"]), []).append(c)
    keep = set()
    for key, cs in out.items():
        pat = W2_MAIN.get(key)
        pool = [c for c in cs if re.fullmatch(pat, c["run_id"])] if pat else []
        pool = pool or cs
        best = max(pool, key=lambda c: (c["collected"] and c["status"] == "done", c["collected"], bool(c["n_steps"]), job_id(c["run_id"])))
        keep.add(best["run_id"])
    return keep


def w2_role(run_id):
    """One W2 series per library in the main charts. TRL's is the sharded-Harbor run; the unsharded local run stays only in
    the step-time chart that explains the fix ("before"); the first run on HF Sandboxes is dropped (footnote only)."""
    if run_id.startswith("trl-w2-M-shared-"):
        return "dropped"
    if re.fullmatch(r"trl-w2-M-local-\d+", run_id):
        return "before_fix"
    if re.fullmatch(r"trl-w2-M-local-sharded-\d+", run_id) or re.fullmatch(r"trl-moe-w2-.*-local-\d+", run_id):
        return "st4_ref"  # TRL at max_staleness 4: a held-out reference only, until the staleness-8 run's evals land
    return "main"


def rd(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def meta_of(run_dir):
    p = os.path.join(run_dir, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def job_id(run_id):
    m = re.search(r"(\d+)$", run_id)
    return int(m.group(1)) if m else 0


def r3(x, n=4):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- run selection


def raw_runs():
    """Every collected run: id -> (dir, meta)."""
    return {os.path.basename(d.rstrip("/")): (d, meta_of(d)) for d in sorted(glob.glob(os.path.join(RAW, "*", "")))}


def is_valid(meta):
    return not meta.get("validity")


def native_trl_steps(run_dir):
    """Steps of a TRL run that isn't collected yet, from its live metrics.jsonl (the same fields collect/trl.py maps)."""
    rows = [r for r in rd(os.path.join(run_dir, "metrics.jsonl")) if "perf/fwd_bwd_s" in r and r.get("perf/fwd_bwd_s") is not None]
    out = []
    for r in rows:
        fb, opt = r.get("perf/fwd_bwd_s"), r.get("perf/optimizer_s")
        fwd, ratio = r.get("batch/forwarded_tokens_per_step"), r.get("batch/packing_ratio")
        out.append({"step": r["step"], "time": r["time"], "step_time": r.get("perf/step_s"),
                    "train_compute_time": fb + opt if fb is not None and opt is not None else None,
                    "wait_for_rollouts_time": r.get("perf/rollout_wait_s"), "weight_sync_time": r.get("perf/weight_sync_s"),
                    "reward_mean": r.get("reward"), "forwarded_tokens": fwd,
                    "trained_tokens": fwd * ratio if fwd is not None and ratio is not None else None})  # fmt: skip
    return out


def pick_run(raw, pattern, native_glob=None):
    """Latest valid collected run matching pattern; else the latest live native TRL dir. -> dict or None."""
    hits = sorted((rid for rid, (_, m) in raw.items() if re.fullmatch(pattern, rid) and is_valid(m)), key=job_id)
    if hits:
        rid = hits[-1]
        d, m = raw[rid]
        steps = sorted(rd(os.path.join(d, "steps.jsonl")), key=lambda s: s["step"])
        return {"run_id": rid, "dir": d, "meta": m, "steps": steps, "eval": rd(os.path.join(d, "eval.jsonl")), "collected": True}
    if native_glob:
        dirs = sorted((d for d in glob.glob(native_glob) if re.fullmatch(pattern, os.path.basename(d))), key=lambda d: job_id(os.path.basename(d)))
        if dirs:
            steps = native_trl_steps(dirs[-1])
            return {"run_id": os.path.basename(dirs[-1]), "dir": dirs[-1], "meta": {}, "steps": steps, "eval": [], "collected": False}
    return None


def excluded_runs(raw):
    out = []
    for rid, (_, m) in raw.items():
        if m.get("validity"):
            out.append({"run_id": rid, "lib": m.get("lib"), "reason": m["validity"].split(":")[0]})
        elif "-recommended" in rid:
            out.append({"run_id": rid, "lib": m.get("lib"), "reason": "recommended arm, not featured"})
    return sorted(out, key=lambda x: x["run_id"])


# ---------------------------------------------------------------- W1


def step_parts(steps, skip):
    """Mean seconds per optimizer step of train / wait / sync / other over steps > skip (rows can span several steps)."""
    rows, prev = [], None
    for s in steps:
        span = s["step"] - (prev if prev is not None else 0)
        prev = s["step"]
        if s["step"] > skip and s.get("step_time") is not None:
            rows.append((s, span))
    if not rows:
        return None
    tot = mean([s["step_time"] / k for s, k in rows])
    parts, logged = {}, {}
    for key, f in [("train", "train_compute_time"), ("wait", "wait_for_rollouts_time"), ("sync", "weight_sync_time")]:
        v = mean([s[f] / k if s.get(f) is not None else None for s, k in rows])
        logged[key] = v is not None
        parts[key] = v or 0.0
    parts["other"] = max(tot - sum(parts.values()), 0.0)
    return {"step_time": r3(tot, 2), **{k: r3(v, 2) for k, v in parts.items()}, "logged": logged, "n_steps": len(rows)}


def w1_section(raw):
    base = {s: (rd(os.path.join(RAW, f"base-Qwen3.5-{m}", "eval.jsonl")) or [None])[0] for s, m in [("S", "2B"), ("M", "9B")]}
    out = {}
    for size, libs in W1_EXPECTED.items():
        b = base[size]
        runs = []
        for lib in libs:
            r = pick_run(raw, rf"{re.escape(lib)}-w1-{size}-shared(-builtin)?-\d+", os.path.join(RUNS, lib, f"{lib}-w1-{size}-shared*"))
            if r is None:
                runs.append({"lib": lib, "status": "in progress", "status_note": "no run collected yet", "evals": [], "steps": [],
                             "pending_evals": W1_CKPTS})  # fmt: skip
                continue
            st = [s for s in r["steps"] if s.get("time") is not None]
            last = st[-1]["step"] if st else 0
            t0 = st[0]["time"] if st else None
            evals = {}
            for e in r["eval"]:
                if e["step"] <= W1_STEPS:  # SkyRL also evaluates its final export as step 201, a duplicate of 200
                    evals[e["step"]] = {"step": e["step"], "aime24": r3(e["aime24_avg16"]), "aime25": r3(e["aime25_avg16"]),
                                        "math500": r3(e["math500_avg1"])}  # fmt: skip
            done = r["collected"] and last >= W1_STEPS and r["meta"].get("status") in (None, "completed")
            pending = [c for c in W1_CKPTS if c not in evals]
            status = "done" if done and not pending else ("evals pending" if done else "in progress")
            note = {"done": "", "evals pending": f"evals of step {', '.join(map(str, pending))} pending",
                    "in progress": f"training, step {last} of {W1_STEPS}"}[status]  # fmt: skip
            runs.append({
                "lib": lib, "run_id": r["run_id"], "status": status, "status_note": note, "collapsed": lib in COLLAPSED,
                "evals": [evals[k] for k in sorted(evals)], "pending_evals": pending,
                "last_step": last, "hours": r3((st[-1]["time"] - t0) / 3600, 3) if st else None,
                "job_hours": r3((r["meta"]["end_time"] - r["meta"]["start_time"]) / 3600, 3)
                if r["meta"].get("end_time") and r["meta"].get("start_time") else None,
                "steps": [{"step": s["step"], "h": r3((s["time"] - t0) / 3600, 4), "reward": r3(s.get("reward_mean")),
                           "step_time": r3(s.get("step_time"), 2)} for s in st],
                "parts": step_parts(r["steps"], SKIP_W1),
            })  # fmt: skip
        extra = []
        if size == "S":  # SkyRL's default loss (dppo) on the shared W1 config: the as-shipped SkyRL data point, next to the collapsed rollout_is mapping
            r = pick_run(raw, r"skyrl-w1-S-recommended-\d+")
            if r is not None:
                extra.append({"lib": "skyrl", "label": "SkyRL default dppo", "run_id": r["run_id"],
                              "evals": [{"step": e["step"], "aime24": r3(e["aime24_avg16"]), "aime25": r3(e["aime25_avg16"]),
                                         "math500": r3(e["math500_avg1"])} for e in sorted(r["eval"], key=lambda e: e["step"]) if e["step"] <= W1_STEPS]})  # fmt: skip
        out[size] = {
            "extra": extra,
            "base": {"aime24": r3(b["aime24_avg16"]), "aime25": r3(b["aime25_avg16"]), "math500": r3(b["math500_avg1"])} if b else None,
            "runs": runs,
            "not_run": [{"lib": k, "reason": v} for k, v in W1_NOT_RUN[size].items()],
        }
    return out


def throughput_S(w1):
    """Fair W1-S throughput rows (collect/throughput_fair.py) of the runs plotted in W1-S."""
    path = os.path.join(BENCH, "results", "tables", "throughput_fair.csv")
    if not os.path.exists(path):
        return []
    rows = {r["run_id"]: r for r in csv.DictReader(open(path))}
    out = []
    for run in w1["S"]["runs"]:
        r = rows.get(run.get("run_id"))
        if r is None:
            out.append({"lib": run["lib"], "run_id": run.get("run_id"), "status": "in progress"})
            continue
        f = lambda k: r3(float(r[k]), 1) if r.get(k) not in (None, "") else None  # noqa: E731
        out.append({"lib": run["lib"], "run_id": run["run_id"], "status": "done", "collapsed": run["collapsed"],
                    "step_time": f("step_time_s"), "all": f("all_trained_tok_s"), "useful": f("useful_trained_tok_s"),
                    "generated": f("generated_tok_s"), "zero_adv_frac": r3(float(r["zero_adv_frac"]), 3),
                    "useful_exact": r["useful_is_exact"] == "True", "padded_forward": f("padded_forward_tok_s")})  # fmt: skip
    return out


# ---------------------------------------------------------------- W2


def lib_of(run_id):
    return next((lib for lib in sorted(LIBS, key=len, reverse=True) if run_id.startswith(lib + "-")), run_id.split("-")[0])


def family_of(run_id, meta=None):
    """9B or 35B-A3B: the W2 model, from the run id or meta.json."""
    text = run_id + " " + str((meta or {}).get("model", ""))
    return "35B-A3B" if ("35B" in text or "-moe-" in text) else "9B"


def w2_label(run_id):
    for prefix, labels in W2_LABEL.items():
        if run_id.startswith(prefix):
            return labels
    lib, fam = lib_of(run_id), family_of(run_id)
    name = LIB_NAME.get(lib, lib) + (f", {fam} LoRA" if fam != "9B" else "")
    return (name, f"{name} ({run_id})")


W2_NATIVE = [  # live run dirs, shown before collection (TRL: from metrics.jsonl; others: an in-progress row until collected)
    os.path.join(RUNS, "trl", "trl-w2-M-*"),
    os.path.join(RUNS, "trl-moe-ep", "*w2*"),
    os.path.join(RUNS, "prime-rl", "prime-rl-w2-moe-*"),
    os.path.join(RUNS, "verl", "verl-w2-*"),
    os.path.join(RUNS, "miles", "miles-w2-*"),
]


def w2_runs(raw):
    """Every W2 training run on built-in options (9B and 35B): collected ones, plus live dirs not collected yet."""
    ids = [rid for rid, (_, m) in raw.items() if "-w2-" in rid and re.search(r"-\d+$", rid) and is_valid(m)
           and "smoke" not in rid and m.get("status") not in ("killed", "failed")]  # fmt: skip
    runs = [pick_run(raw, re.escape(rid)) for rid in ids]
    for pattern in W2_NATIVE:
        for d in glob.glob(pattern):
            rid = os.path.basename(d.rstrip("/"))
            if rid in raw or not re.search(r"-\d+$", rid) or any(x in rid for x in ("smoke", "probe", "eval", "-aux")):
                continue
            r = {"run_id": rid, "steps": native_trl_steps(d), "collected": False, "dir": d, "meta": {}}
            runs.append(r)
    return sorted(runs, key=lambda r: (family_of(r["run_id"], r["meta"]) != "9B", lib_of(r["run_id"]) != "trl", job_id(r["run_id"])))


def w2_section(raw):
    breakdown, prefix, reward, dropped, st4 = [], [], [], [], []
    runs = w2_runs(raw)
    cands = [{"run_id": r["run_id"], "lib": lib_of(r["run_id"]), "family": family_of(r["run_id"], r["meta"]), "collected": r["collected"],
              "status": "done" if r["collected"] and r["meta"].get("status") == "completed" else "x", "n_steps": len(r["steps"])}
             for r in runs if w2_role(r["run_id"]) == "main"]  # fmt: skip
    main_keep = pick_main(cands)
    for r in runs:
        lib = lib_of(r["run_id"])
        role = w2_role(r["run_id"])
        if role == "main" and r["run_id"] not in main_keep:
            continue  # another run stands for this library (ablations, superseded attempts)
        if role == "st4_ref":
            st4.append(r["run_id"])
            continue
        if role == "dropped":
            p = step_parts(r["steps"], SKIP_W2)
            dropped.append({"run_id": r["run_id"], "step_time": p["step_time"] if p else None})
            continue
        short, long = w2_label(r["run_id"])
        last = r["steps"][-1]["step"] if r["steps"] else 0
        status = "done" if r["collected"] and r["meta"].get("status") == "completed" else "in progress"
        base = {"lib": lib, "run_id": r["run_id"], "label": short, "long_label": long, "status": status, "last_step": last,
                "family": family_of(r["run_id"], r["meta"]),
                "status_note": "" if status == "done" else (f"training, step {last}" if last else "starting, or not collected yet")}  # fmt: skip
        base["role"] = role
        breakdown.append(base | {"parts": step_parts(r["steps"], SKIP_W2)})
        if role != "main":
            continue
        reward.append(base | {"steps": [{"step": s["step"], "reward": r3(s.get("reward_mean"))} for s in r["steps"] if s.get("reward_mean") is not None]})
        ss = [s for s in r["steps"] if s["step"] > SKIP_W2 and s.get("forwarded_tokens") and s.get("trained_tokens")]
        prefix.append(base | {
            "trained": r3(mean([s["trained_tokens"] for s in ss]), 0), "forwarded": r3(mean([s["forwarded_tokens"] for s in ss]), 0),
            "per_step": [{"step": s["step"], "trained": round(s["trained_tokens"]), "forwarded": round(s["forwarded_tokens"])} for s in ss],
        })  # fmt: skip
        if prefix[-1]["trained"] and prefix[-1]["forwarded"]:
            prefix[-1]["ratio"] = r3(prefix[-1]["trained"] / prefix[-1]["forwarded"], 2)
    main_ids = [b["run_id"] for b in breakdown if b["role"] == "main"]
    # Failed W2 runs: the latest failed run of a library (and model) that has no working run, as a labelled row.
    last_fail = {}
    for rid, (_, m) in raw.items():
        if "-w2-" in rid and "smoke" not in rid and m.get("status") == "failed" and is_valid(m):
            key = (lib_of(rid), family_of(rid, m))
            last_fail[key] = max(last_fail.get(key, 0), job_id(rid))
    # A live dir with no steps that is older than the library's latest failed run is a stale attempt: the failure stands.
    stale = {b["run_id"] for b in breakdown if b["role"] == "main" and not b.get("parts") and b["last_step"] == 0
             and job_id(b["run_id"]) < last_fail.get((b["lib"], b["family"]), 0)}  # fmt: skip
    breakdown = [b for b in breakdown if b["run_id"] not in stale]
    prefix = [b for b in prefix if b["run_id"] not in stale]
    reward = [b for b in reward if b["run_id"] not in stale]
    main_ids = [i for i in main_ids if i not in stale]
    have = {(b["lib"], b["family"]) for b in breakdown if b["role"] == "main"}
    failed = {}
    for rid, (_, m) in sorted(raw.items(), key=lambda kv: job_id(kv[0])):
        if "-w2-" in rid and "smoke" not in rid and m.get("status") == "failed" and is_valid(m):
            key = (lib_of(rid), family_of(rid, m))
            if key not in have:
                kind = m.get("failure_kind")
                failed[key] = {"lib": key[0], "family": key[1], "run_id": rid, "kind": kind,
                               "note": FAIL_SHORT.get(kind, f"failed ({kind})" if kind else "failed"),
                               "detail": m.get("failure_reason") or m.get("failure") or ""}  # fmt: skip
    for key, note in W2_NOT_RUN.items():
        if key not in have and key not in failed:
            failed[key] = {"lib": key[0], "family": key[1], "run_id": None, "kind": "not_run", "note": note, "detail": W2_NOT_RUN_DETAIL.get(key, "")}
    failed = sorted(failed.values(), key=lambda f: (f["family"] != "9B", LIBS.index(f["lib"]) if f["lib"] in LIBS else 99))
    return {"step_breakdown": breakdown, "prefix": prefix, "reward": reward, "dropped": dropped, "failed": failed, "waterfall": w2_waterfall(),
            "episode": w2_episode(), "heldout": w2_heldout(main_ids, st4)}  # fmt: skip


def w2_waterfall():
    """The counterfactual chain of envs/trl/W2_STEP_TIME.md (prime-rl's step -> TRL's), parsed from the doc so a revision
    of the analysis flows into the chart."""
    md = open(os.path.join(BENCH, "envs", "trl", "W2_STEP_TIME.md")).read()
    m = re.search(r"from prime-rl's (\d+) s to TRL local's (\d+) s", md)
    measured = re.search(r"against (\d+) s predicted and (\d+) s measured", md)
    causes = []
    for n, (label, short) in enumerate([
        ("The Harbor server's single event loop", "Harbor server: one event loop for all 64 agents"),
        ("Box lifecycle outside the agent", "Box start, setup and verifier"),
        ("Episodes per step", "Bigger steps (refilled to a token budget)"),
        ("Budget tail", "30-call cap never reached the agent"),
    ], 1):  # fmt: skip
        c = re.search(rf"{n}\. \*\*{re.escape(label)}: \+(\d+) s/step", md)
        causes.append({"label": short, "doc_label": label, "s": int(c.group(1)) if c else None})
    return {"source": "envs/trl/W2_STEP_TIME.md", "before_run": "trl-w2-M-local-366992", "start": int(m.group(1)), "predicted": int(m.group(2)),
            "measured": int(measured.group(2)) if measured else None, "causes": causes}  # fmt: skip


def w2_episode():
    """Mean episode wall split into phases, from stable-bench-runs/status/w2_step_time.json (steady, scored episodes)."""
    p = os.path.join(RUNS, "status", "w2_step_time.json")
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    t, pr = d["trl_local"], d["prime"]
    trl = {"vllm": t["vllm_s_per_episode_est"], "agent": t["agent_exec"]["mean"] - t["vllm_s_per_episode_est"],
           "box": t["env_setup"]["mean"] + t["agent_setup"]["mean"], "verifier": t["verifier"]["mean"], "outside": t["wall_outside_phases"]}  # fmt: skip
    prime = {"vllm": pr["agent_model"]["mean"], "agent": pr["agent_harness"]["mean"], "box": pr["boot"]["mean"] + pr["setup"]["mean"],
             "verifier": pr["scoring"]["mean"], "outside": 0.0}  # fmt: skip
    return [
        {"lib": "trl", "run_id": t["run"], "label": "TRL before the Harbor fix", "wall": t["wall"]["mean"],
         "parts": {k: r3(v, 1) for k, v in trl.items()}, "timeout_frac": 0.47, "calls": t["turns"]["mean"]},
        {"lib": "prime-rl", "run_id": pr["run"], "label": "prime-rl", "wall": pr["wall"]["mean"],
         "parts": {k: r3(v, 1) for k, v in prime.items()}, "timeout_frac": 0.0, "calls": pr["turns"]["mean"]},
    ]  # fmt: skip


def w2_heldout(main_ids, st4=()):
    """Held-out W2 evals (14 tasks x 4 or x 12 episodes) from every finished probe eval dir; EVAL_MODEL says which checkpoint."""
    points = []
    for d in sorted(glob.glob(os.path.join(RUNS, "trl", "trl-w2probe-*-eval-*"))):
        env = dict(line.strip().split("=", 1) for line in open(os.path.join(d, "launch.env")) if "=" in line)
        model = env.get("EVAL_MODEL", "")
        log = open(os.path.join(d, "probe-ckpt.log")).read() if os.path.exists(os.path.join(d, "probe-ckpt.log")) else ""
        eps = rd(os.path.join(d, "probe-ckpt.jsonl"))
        done = "mean reward" in log
        scored = [e for e in eps if e.get("reward") is not None]
        # Checkpoint layouts: <run>/checkpoint-N (TRL), <run>/hf/step_N (prime-rl), <run>/ckpt/global_step_N/... (verl),
        # merged/<run>-checkpoint-N (TRL MoE LoRA merged for eval).
        run = (re.search(r"/([^/]+?)-checkpoint-(\d+)/?$", model) or re.search(r"/([^/]+)/(?:checkpoint-|hf/step_|step_)(\d+)", model)
               or re.search(r"/([^/]+)/ckpt/global_step_(\d+)", model))  # fmt: skip
        step = run
        family = family_of(run.group(1)) if run else ("35B-A3B" if "35B" in model else "9B")
        p = {"eval_job": os.path.basename(d), "model": model, "family": family, "split": env.get("SPLIT", "eval"),
             "run_id": run.group(1) if run else None, "step": int(run.group(2)) if run else 0,
             "status": "done" if done else "in progress", "n": len(scored), "unscored": len(eps) - len(scored),
             "samples": 12 if "x12" in env.get("TAG", "") or env.get("SAMPLES") == "12" else int(env.get("SAMPLES", 4))}  # fmt: skip
        if done and scored:
            m = sum(e["reward"] for e in scored) / len(scored)
            p["mean"] = r3(m)
            p["se"] = r3(math.sqrt(m * (1 - m) / len(scored)))
            # Episodes of one task are correlated: SE of the mean over per-task means (14 tasks) for comparison.
            by_task = {}
            for e in scored:
                by_task.setdefault(e["task"], []).append(e["reward"])
            tm = [sum(v) / len(v) for v in by_task.values()]
            mu = sum(tm) / len(tm)
            p["se_task"] = r3(math.sqrt(sum((x - mu) ** 2 for x in tm) / (len(tm) - 1) / len(tm))) if len(tm) > 1 else None
            p["tasks"] = len(tm)
        points.append(p)
    # Two evals of one checkpoint (a rerun): keep the finished one, then the 12-sample one, then the latest job.
    best = {}
    for p in sorted(points, key=lambda p: (p["status"] == "done", p["samples"] == 12, job_id(p["eval_job"]))):
        best[(p["run_id"] or p["model"], p["split"], p["step"])] = p
    # Only the W2 held-out split (SPLIT=eval) goes in this chart; other splits (TermGrade-hard) never mix with it.
    other = [p for p in best.values() if p["split"] != "eval"]
    points = [p for p in best.values() if p["split"] == "eval"]
    series = {}
    for p in points:
        if p["run_id"]:
            lib = lib_of(p["run_id"])
            s = series.setdefault(p["run_id"], {"run_id": p["run_id"], "lib": lib, "family": p["family"], "label": w2_label(p["run_id"])[0], "points": []})
            s["points"].append(p)
    keep = {k: v for k, v in series.items() if k in main_ids}  # one series per library: no older TRL runs
    for rid in main_ids:  # a main run with no eval yet still gets its (empty, in progress) series
        if rid not in keep:
            keep[rid] = {"run_id": rid, "lib": lib_of(rid), "family": family_of(rid), "label": w2_label(rid)[0], "points": []}
    # TRL at staleness 4: shown as a reference series only while the staleness-8 run of that model has no finished eval.
    notes = []
    for rid in st4:
        fam = family_of(rid)
        st8_done = any(s["lib"] == "trl" and s["family"] == fam and any(p["status"] == "done" for p in s["points"]) for s in keep.values())
        ref = series.get(rid)
        if ref is None:
            continue
        if st8_done:
            done = [p for p in ref["points"] if p["status"] == "done"]
            notes.append(f"Not shown: TRL at max_staleness 4 on Qwen3.5-{fam} (run {job_id(rid)}): "
                         + ", ".join(f"step {p['step']} {p['mean']:.3f}" for p in sorted(done, key=lambda p: p["step"])) + ".")
        else:
            keep[rid] = ref | {"label": "TRL, staleness 4", "reference": True}
    series = keep
    for s in series.values():
        s["points"].sort(key=lambda p: p["step"])
        s["pending"] = [c for c in (20, 40, 60) if c not in [p["step"] for p in s["points"] if p["status"] == "done"]]
    base = [p for p in points if not p["run_id"]]
    return {"base": base, "series": sorted(series.values(), key=lambda s: (s["lib"] != "trl", s.get("reference", False), job_id(s["run_id"]))),
            "episodes_per_eval": max((p["n"] + p["unscored"] for p in points if p["status"] == "done"), default=56), "notes": notes,
            "other_splits": [{k: p.get(k) for k in ("eval_job", "split", "model", "family", "run_id", "step", "status", "mean", "se", "n")} for p in other]}  # fmt: skip


# ---------------------------------------------------------------- MoE


def moe_section():
    dirs = sorted(glob.glob(os.path.join(RUNS, "trl-moe-ep", "*-w1-200-*")), key=lambda d: job_id(d))
    if not dirs:
        return None
    d = dirs[-1]
    rows = [r for r in rd(os.path.join(d, "metrics.jsonl")) if r.get("perf/fwd_bwd_s") is not None]
    mem = {}
    for f in glob.glob(os.path.join(d, "mem-rank*.jsonl")):
        for m in rd(f):
            cur = mem.setdefault(m["step"], {"alloc": 0.0, "reserved": 0.0})
            cur["alloc"] = max(cur["alloc"], m["peak_allocated_gb"])
            cur["reserved"] = max(cur["reserved"], m["peak_reserved_gb"])
    steps = []
    for r in rows:
        steps.append({
            "step": r["step"], "step_time": r3(r.get("perf/step_s"), 2), "wait": r3(r.get("perf/rollout_wait_s"), 2),
            "fwd_bwd": r3(r.get("perf/fwd_bwd_s"), 2), "optimizer": r3(r.get("perf/optimizer_s"), 2),
            "sync": r3(r.get("perf/weight_sync_s"), 3), "zero_std": r3(r.get("frac_reward_zero_std"), 3),
            "reward": r3(r.get("reward"), 3), "completion_len": r3(r.get("completions/mean_length"), 0),
            "logp_diff": r3(r.get("sampling/sampling_logp_difference/mean"), 5),
            "peak_alloc_gb": r3(mem.get(r["step"], {}).get("alloc"), 2), "peak_reserved_gb": r3(mem.get(r["step"], {}).get("reserved"), 2),
        })  # fmt: skip
    hours = (rows[-1]["time"] - rows[0]["time"]) / 3600 if rows else None
    w2 = sorted(os.path.basename(x) for x in glob.glob(os.path.join(RUNS, "trl-moe-ep", "*w2*")))
    return {"run_id": os.path.basename(d), "hours": r3(hours, 2), "steps": steps, "w2_runs_seen": w2}


# ---------------------------------------------------------------- text-to-SQL 35B (collect/t2sql_35b.py)


def t2sql_section():
    """Text-to-SQL, Qwen3.5-35B-A3B LoRA: greedy evals per checkpoint (with hours and GPU-hours since job start) and the
    per-step training series, from results/tables/t2sql_35b.csv and t2sql_35b_steps.csv."""
    path = os.path.join(BENCH, "results", "tables", "t2sql_35b.csv")
    if not os.path.exists(path):
        return None
    num = lambda v: None if v in ("", None) else float(v)  # noqa: E731
    evals = [{k: (v if k in ("lib", "run_id") else num(v)) for k, v in r.items()} for r in csv.DictReader(open(path))]
    steps = {}
    for r in csv.DictReader(open(os.path.join(BENCH, "results", "tables", "t2sql_35b_steps.csv"))):
        steps.setdefault(r["lib"], []).append({"step": int(r["step"]), "h": r3(num(r["hours"]), 4), "step_s": r3(num(r["step_s"]), 1),
                                               "wait_s": r3(num(r["wait_s"]), 1), "reward": r3(num(r["reward"]))})  # fmt: skip
    base = next((e for e in evals if e["lib"] == "base"), None)
    runs = []
    for lib in ("trl-fixed", "trl", "prime-rl", "skyrl"):
        ev = [e for e in evals if e["lib"] == lib]
        st = steps.get(lib, [])
        gpus = int(ev[0]["gpus"]) if ev else None
        runs.append({"lib": lib, "run_id": ev[0]["run_id"] if ev else None, "gpus": gpus, "last_step": st[-1]["step"] if st else 0,
                     "evals": [{k: e[k] for k in ("step", "hours", "gpu_hours", "episodes_trained", "validation", "validation_se",
                                                   "arcwise", "arcwise_se", "hard", "hard_se")} for e in ev],
                     "steps": st})  # fmt: skip
    return {"base": {k: base[k] for k in ("validation", "validation_se", "arcwise", "arcwise_se", "hard", "hard_se")} if base else None,
            "runs": runs, "planned_steps": 100}  # fmt: skip


# ---------------------------------------------------------------- robustness


def robustness():
    rows = rd(os.path.join(BENCH, "results", "robustness", "failures.jsonl"))
    out = []
    for lib in LIBS:
        mine = [r for r in rows if r.get("lib") == lib]
        fails = [r for r in mine if r.get("kind") == "failure"]
        cats = {}
        for r in fails:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        out.append({
            "lib": lib, "failures": len(fails),
            **{s: sum(r.get("severity") == s for r in fails) for s in ("blocker", "major", "minor")},
            "unattributed": sum(r.get("kind") == "failure_unattributed" for r in mine),
            "silent": cats.get("silent_correctness", 0), "categories": cats,
            "rows": [{"id": r["id"], "severity": r["severity"], "category": r["category"], "symptom": r["symptom"][:260]} for r in fails],
        })  # fmt: skip
    return {"n_rows": len(rows), "libs": out}


# ---------------------------------------------------------------- vLLM pause repro (no results/raw dir)

# envs/trl/PAUSE_STALL.md "Options measured", standalone vLLM 0.33 repro job 365055 (8 API servers, 688 in flight,
# /pause + /resume every 20 s, 600 s per arm). Same numbers as charts.py's l_pause_options.
PAUSE = [("top-8 (default)", 12.0, 38.2, 65.3), ("top-4", 6.7, 19.3, 79.9), ("top-2", 0.05, 0.21, 91.1),
         ("top-1", 0.04, 0.06, 91.9), ("no logprobs", 0.05, 0.07, 95.9)]  # fmt: skip


def main():
    raw = raw_runs()
    w1 = w1_section(raw)
    data = {
        "generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "lib_names": LIB_NAME,
        "w1": w1,
        "throughput_S": throughput_S(w1),
        "w2": w2_section(raw),
        "moe": moe_section(),
        "t2sql": t2sql_section(),
        "robustness": robustness(),
        "pause": [{"option": o, "pause_mean_s": a, "pause_max_s": b, "gen_ktok_s": c} for o, a, b, c in PAUSE],
        "excluded": excluded_runs(raw),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB)")
    for size in ("S", "M"):
        for r in w1[size]["runs"]:
            print(f"W1-{size} {r['lib']:9s} {r.get('run_id', '-'):40s} {r['status']:14s} {r['status_note']}")
    for r in data["w2"]["step_breakdown"]:
        print(f"W2 {r['run_id']:40s} {r['status']:12s} {r['parts']}")
    for s in data["w2"]["heldout"]["series"]:
        print("W2 eval", s["run_id"], [(p["step"], p.get("mean"), p.get("se"), p.get("se_task")) for p in s["points"]], "pending", s["pending"])
    print("W2 eval base", [(p["model"], p.get("mean"), p.get("se")) for p in data["w2"]["heldout"]["base"]])
    print("waterfall", data["w2"]["waterfall"])
    for r in (data["t2sql"] or {}).get("runs", []):
        print("t2sql", r["lib"], "step", r["last_step"], [(e["step"], e["hours"], e["validation"], e["arcwise"]) for e in r["evals"]])
    print("robustness", [(x["lib"], x["failures"], x["blocker"], x["major"], x["minor"], x["unattributed"]) for x in data["robustness"]["libs"]])


if __name__ == "__main__":
    main()
