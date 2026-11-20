"""Text-to-SQL, Qwen3.5-35B-A3B LoRA: TRL vs prime-rl vs SkyRL, per evaluated checkpoint and per training step.

    python collect/t2sql_35b.py      # in a hopper-cpu job (figures/build.sh runs it); idempotent

Writes results/tables/t2sql_35b.md, t2sql_35b.csv (one row per evaluated checkpoint, plus the base model) and
t2sql_35b_steps.csv (one row per training step). Every checkpoint with a summary.json under t2sql-eval/ is picked up,
so rerun it as evals land.

Sources: greedy evals t2sql-eval/<run>-checkpoint-<N>/summary.json (TRL's eval.sbatch, shared); job start and GPU
count from sacct; per step TRL metrics.jsonl, prime-rl <run>-aux/steps.csv, SkyRL tb/events.* (read with a small
TFRecord parser below, so no tensorboard install is needed).
"""

import calendar
import csv
import glob
import json
import os
import re
import struct
import subprocess
import time


RUNS = os.environ.get("RUNS", "/fsx/amine_dirhoussi/stable-bench-runs")
EVAL = os.path.join(RUNS, "trl-moe-ep", "t2sql-eval")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables", "t2sql_35b")
SPLITS = ["validation", "arcwise", "hard"]
LIBS = {
    "trl": {"name": "TRL", "run": "trl-t2sql-Qwen3.5-35B-A3B-ep4-lora32-main-373132", "job": 373132},
    "trl-fixed": {"name": "TRL (with #8212 fix)", "run": "trl-t2sql-Qwen3.5-35B-A3B-ep4-lora32-salt-374011", "job": 374011},
    "prime-rl": {"name": "prime-rl", "run": "prime-rl-t2sql-Qwen3.5-35B-A3B-lora32-norefill-389085", "job": 389085},
    "prime-rl-refill": {"name": "prime-rl, refill", "run": "prime-rl-t2sql-Qwen3.5-35B-A3B-lora32-373642", "job": 373642},
    "skyrl": {"name": "SkyRL", "run": "skyrl-t2sql-Qwen3.5-35B-A3B-lora32-373788", "job": 373788},
}
# From workloads/text2sql/README.md item 8, envs/prime-rl/NOTES.md "Text-to-SQL 35B" item 3, envs/skyrl/NOTES.md
# "35B text-to-SQL" items 5-6.
SETUP = {
    "trl": "clipped GRPO (eps 0.2 / 0.28), no KL, mean-centred advantage; lr 5e-5, 10 warmup steps, cosine to 10%; "
           "LoRA r32 / alpha 64 on attention + GDN + shared expert (310 modules); batch = token budget 32,768 x 10 "
           "micro-batches per rank (mean ~57 groups, ~907 samples per step, zero-std groups dropped); 16 samples per "
           "question; T = 1.0; async, max staleness 8, 256 episodes in flight; 8 GPUs (trainer EP4 x FSDP4 on 4, vLLM DP4 + EP on 4)",
    "trl-fixed": "same settings as TRL 373132, plus a per-request vLLM cache_salt of the weight version (the TRL #8212 fix: "
                 "vLLM was paused with clear_cache=false, so the shared prompt prefix kept base-weight KV all run)",
    "prime-rl": "same settings as prime-rl 373642 except the batch: 960 samples per step (60 x 16), zero-advantage "
                "groups dropped and not refilled (constant_trainer_batch_size = false), the same rule as TRL",
    "prime-rl-refill": "prime-rl's IPO loss (masked where |pi - mu| > 0.3, no ratio clip), no KL, mean-centred advantage; "
                "lr 5e-5, 10 warmup steps, cosine to 10%; LoRA r32 / alpha 64 on the same 310 modules; batch 912 samples "
                "(57 x 16), zero-advantage samples pruned and refilled; 16 samples per question; T = 1.0; async, "
                "max staleness 8, 256 episodes in flight; 8 GPUs (trainer EP4 x FSDP4 on 4, vLLM DP4 on 4)",
    "skyrl": "SkyRL-SQL Megatron LoRA recipe: dppo loss (main's default), no KL, mean-centred advantage; lr 3e-5 constant; "
             "LoRA r32 / alpha 64; batch 256 prompts x 5 samples = 1,280 episodes, one optimizer step; 5 samples per "
             "question; T = 0.6, top_p 0.95; synchronous (staleness 0), colocated; 4 GPUs (Megatron TP2 EP4 DP2 + one vLLM "
             "TP4; the recipe assumes 1 x 8, 4 was what was free)",
}  # fmt: skip
PRIME_BATCH = {"prime-rl": 960, "prime-rl-refill": 912}  # samples generated per step (refill: trained per step)
SKYRL_BATCH = 1280  # SkyRL: 256 prompts x 5, every episode trained


def sacct(job):
    """(start unix time, GPUs) of a job."""
    out = subprocess.run(["sacct", "-nX", "-P", "-j", str(job), "-o", "Start,AllocTRES"], capture_output=True, text=True, check=True).stdout
    start, tres = out.strip().splitlines()[0].split("|")
    return calendar.timegm(time.strptime(start, "%Y-%m-%dT%H:%M:%S")), int(re.search(r"gres/gpu=(\d+)", tres).group(1))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------- per-step readers -> [{step, t, step_s, wait_s, out_tokens, turns, reward, trained, generated}]


def trl_steps(run):
    d = os.path.join(RUNS, "trl-moe-ep", run)
    rows = [json.loads(line) for line in open(os.path.join(d, "metrics.jsonl"))]
    rows = [r for r in rows if "train_runtime" not in r]  # the last row is the run summary
    ends = sorted(json.loads(line)["end"] for line in open(os.path.join(d, "episodes.jsonl")) if '"evaluation": false' in line)
    out, trained, k = [], 0, 0
    for r in rows:
        trained += r["batch/samples_per_step"] / r["rollout/samples_per_rollout"]
        while k < len(ends) and ends[k] <= r["time"]:
            k += 1
        out.append({"step": r["step"], "t": r["time"], "step_s": r.get("perf/step_s"), "wait_s": r["perf/rollout_wait_s"],
                    "out_tokens": r["completions/mean_length"], "turns": r["rollout/turns_mean"], "reward": r["rewards/rollout/mean"],
                    "trained": round(trained), "generated": k})  # fmt: skip
    return out


def prime_steps(run, batch, refill):
    """Refill trains `batch` samples every step; without it, the zero-advantage samples prime-rl's own metrics count as
    pruned are dropped from the step."""
    pruned = {}
    if not refill:
        for line in open(os.path.join(RUNS, "prime-rl", run, "monitors", "file", "metrics.jsonl")):
            r = json.loads(line)
            if "progress/output_tokens" in r:
                pruned[r["step"]] = r.get("rollout/queued/pruned") or 0
    out, gen, trained = [], 0, 0
    for r in csv.DictReader(open(os.path.join(RUNS, "prime-rl", run + "-aux", "steps.csv"))):
        gen += int(r["rollouts"])
        n = int(r["step"])
        trained += batch - pruned.get(n, 0)
        out.append({"step": n, "t": float(r["time"]), "step_s": float(r["step_s"]), "wait_s": float(r["rollout_wait_s"]),
                    "out_tokens": float(r["out_tokens_mean"]), "turns": float(r["turns_mean"]), "reward": float(r["reward"]),
                    "trained": int(trained), "generated": gen})  # fmt: skip
    return out


def _varint(b, i):
    shift = val = 0
    while True:
        c = b[i]
        i += 1
        val |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return val, i


def _fields(b):
    """Protobuf wire fields of one message: (field number, raw value)."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        wire = key & 7
        if wire == 0:
            v, i = _varint(b, i)
        elif wire == 1:
            v, i = b[i : i + 8], i + 8
        elif wire == 5:
            v, i = b[i : i + 4], i + 4
        else:
            n, i = _varint(b, i)
            v, i = b[i : i + n], i + n
        yield key >> 3, v


def tb_scalars(path):
    """{step: {tag: value, "_t": last wall time}} from a TensorBoard event file of simple_value scalars."""
    data, i, out = open(path, "rb").read(), 0, {}
    while i < len(data):
        n = struct.unpack("<Q", data[i : i + 8])[0]
        ev, i = data[i + 12 : i + 12 + n], i + 16 + n
        wt, step, vals = None, 0, []
        for f, v in _fields(ev):
            if f == 1:
                wt = struct.unpack("<d", v)[0]
            elif f == 2:
                step = v
            elif f == 5:
                for _, val in _fields(v):
                    kv = dict(_fields(val))
                    if 1 in kv and 2 in kv:
                        vals.append((kv[1].decode(), struct.unpack("<f", kv[2])[0]))
        if vals:
            d = out.setdefault(step, {})
            d.update(vals)
            d["_t"] = max(d.get("_t", 0), wt)
    return out


def skyrl_steps(run):
    tb = tb_scalars(sorted(glob.glob(os.path.join(RUNS, "skyrl", run, "tb", "events.*")))[-1])
    out = []
    for n in sorted(tb):
        s = tb[n]
        if "timing/step" not in s or "reward/avg_raw_reward" not in s:
            continue  # a step whose metrics are still being written
        out.append({"step": n, "t": s["_t"], "step_s": s["timing/step"], "wait_s": s["timing/generate"],
                    "out_tokens": s.get("generate/avg_assistant_tokens"), "turns": s.get("environment/turns"),
                    "reward": s["reward/avg_raw_reward"], "trained": n * SKYRL_BATCH, "generated": n * SKYRL_BATCH})  # fmt: skip
    return out


READERS = {"trl": trl_steps, "trl-fixed": trl_steps, "skyrl": skyrl_steps,
           "prime-rl": lambda run: prime_steps(run, PRIME_BATCH["prime-rl"], refill=False),
           "prime-rl-refill": lambda run: prime_steps(run, PRIME_BATCH["prime-rl-refill"], refill=True)}


# ---------------------------------------------------------------- evals


def evals():
    """{(run or "base", step): {split: (pass_rate, se, episodes)}} from every finished summary.json."""
    out = {}
    for p in sorted(glob.glob(os.path.join(EVAL, "*", "summary.json"))):
        name = os.path.basename(os.path.dirname(p))
        m = re.fullmatch(r"(.+)-checkpoint-(\d+)", name)
        key = (m.group(1), int(m.group(2))) if m else ("base", 0)
        s = json.load(open(p))
        out[key] = {sp: (s[f"eval-{sp}"]["pass_rate"], s[f"eval-{sp}"]["pass_rate_se"], s[f"eval-{sp}"]["episodes"])
                    for sp in SPLITS if f"eval-{sp}" in s}  # fmt: skip
    return out


def fmt(x, n=3):
    return "" if x is None else f"{x:.{n}f}"


def main():
    E = evals()
    rows, step_rows, bins, info = [], [], {}, {}
    b = E.get(("base", 0), {})
    rows.append({"lib": "base", "run_id": "Qwen/Qwen3.5-35B-A3B", "gpus": "", "step": 0, "hours": 0.0, "gpu_hours": 0.0,
                 "episodes_trained": 0, "episodes_generated": 0,
                 **{k: v for sp in SPLITS if sp in b for k, v in ((sp, b[sp][0]), (sp + "_se", b[sp][1]))}})  # fmt: skip
    for lib, L in LIBS.items():
        start, gpus = sacct(L["job"])
        steps = READERS[lib](L["run"])
        info[lib] = {"start": start, "gpus": gpus, "last_step": steps[-1]["step"] if steps else 0}
        by = {s["step"]: s for s in steps}
        for s in steps:
            h = (s["t"] - start) / 3600
            step_rows.append({"lib": lib, "step": s["step"], "hours": round(h, 4), "gpu_hours": round(h * gpus, 3),
                              **{k: (round(s[k], 4) if isinstance(s[k], float) else s[k]) for k in
                                 ("step_s", "wait_s", "out_tokens", "turns", "reward", "trained", "generated")}})  # fmt: skip
        for (run, step), ev in sorted(E.items(), key=lambda kv: kv[0][1]):
            if run != L["run"]:
                continue
            s = by.get(step)
            h = (s["t"] - start) / 3600 if s else None
            rows.append({"lib": lib, "run_id": run, "gpus": gpus, "step": step, "hours": h, "gpu_hours": h * gpus if h is not None else None,
                         "episodes_trained": s["trained"] if s else None, "episodes_generated": s["generated"] if s else None,
                         **{k: v for sp in SPLITS if sp in ev for k, v in ((sp, ev[sp][0]), (sp + "_se", ev[sp][1]))}})  # fmt: skip
        # 10-step bins; step 1 is left out of the time columns (it includes startup and compile)
        for lo in range(1, info[lib]["last_step"] + 1, 10):
            q = [s for s in steps if lo <= s["step"] < lo + 10]
            qt = [s for s in q if s["step"] > 1]
            bins.setdefault(lib, []).append({
                "steps": f"{q[0]['step']}-{q[-1]['step']}", "step_s": mean([s["step_s"] for s in qt]),
                "wait_share": sum(s["wait_s"] for s in qt) / sum(s["step_s"] for s in qt) if qt else None,
                "out_tokens": mean([s["out_tokens"] for s in q]), "turns": mean([s["turns"] for s in q]), "reward": mean([s["reward"] for s in q])})  # fmt: skip

    cols = ["lib", "run_id", "gpus", "step", "hours", "gpu_hours", "episodes_trained", "episodes_generated"] + [x for sp in SPLITS for x in (sp, sp + "_se")]
    with open(OUT + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(OUT + "_steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(step_rows[0]))
        w.writeheader()
        w.writerows(step_rows)

    gen = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    md = [f"# Text-to-SQL, Qwen3.5-35B-A3B LoRA: TRL vs prime-rl vs SkyRL", "",
          f"Generated {gen} by `collect/t2sql_35b.py` (rerun it as checkpoints and evals land). Greedy evals with TRL's "
          "`eval.sbatch` (T=0, 1 sample per task, outcome-only score) for every model; SE is the binomial standard error. "
          "Hard overlaps the train split, so validation and arcwise are the held-out numbers.", "",
          "Hours: wall clock from the job's sacct start to the step's log time (includes startup: model load, vLLM, first rollouts). "
          "GPU-hours: hours x the job's GPUs. Episodes trained: samples that went into optimizer steps (one per episode in all three). "
          "Episodes generated: rollouts finished by then (TRL from episodes.jsonl; prime-rl drops zero-advantage groups, and its "
          "refill run regenerates them; SkyRL trains everything it generates).", "", "## Setup", ""]  # fmt: skip
    for lib, L in LIBS.items():
        md.append(f"- **{L['name']}** (job {L['job']}, {info[lib]['gpus']} x B200, at step {info[lib]['last_step']}): {SETUP[lib]}.")
    md += ["", "## Per checkpoint", "",
           "| library | step | hours | GPU-hours | episodes trained | episodes generated | validation (398) | arcwise (498) | hard (658, in train) |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    name = {"base": "base", **{k: v["name"] for k, v in LIBS.items()}}
    for r in rows:
        sc = " | ".join(f"{fmt(r.get(sp))} ± {fmt(r.get(sp + '_se'))}" if r.get(sp) is not None else "" for sp in SPLITS)
        n = lambda v: "" if v is None else f"{v:,}"  # noqa: E731
        md.append(f"| {name[r['lib']]} | {r['step']} | {fmt(r['hours'], 2)} | {fmt(r['gpu_hours'], 1)} | {n(r['episodes_trained'])} | {n(r['episodes_generated'])} | {sc} |")
    md += ["", "## Per training step, 10-step bins", "",
           "Step time and rollout share leave out step 1 (startup). Rollout share: time the trainer waited for rollouts / step "
           "time (TRL, prime-rl: async); for SkyRL (synchronous) it is generation time / step time. Output tokens: sampled tokens "
           "per episode (TRL `completions/mean_length`, prime-rl `out_tokens_mean`, SkyRL `generate/avg_assistant_tokens`). "
           "Train reward is each library's rollout reward with its own batch (16 samples per question in TRL and prime-rl, 5 in SkyRL).", ""]  # fmt: skip
    for lib, B in bins.items():
        md += [f"**{LIBS[lib]['name']}**", "", "| steps | step time (s) | rollout share | output tokens | turns | train reward |", "|---|---|---|---|---|---|"]
        for x in B:
            share = "" if x["wait_share"] is None else f"{x['wait_share']:.0%}"
            md.append(f"| {x['steps']} | {fmt(x['step_s'], 0)} | {share} | {fmt(x['out_tokens'], 0)} | {fmt(x['turns'], 2)} | {fmt(x['reward'])} |")
        md.append("")
    with open(OUT + ".md", "w") as f:
        f.write("\n".join(md))
    print(f"wrote {OUT}.md, .csv, _steps.csv: {len(rows)} checkpoint rows, {len(step_rows)} step rows")
    for r in rows:
        print(r["lib"], r["step"], fmt(r["hours"], 2), fmt(r["gpu_hours"], 1), r["episodes_trained"], r["episodes_generated"],
              *(fmt(r.get(sp)) for sp in SPLITS))  # fmt: skip


if __name__ == "__main__":
    main()
