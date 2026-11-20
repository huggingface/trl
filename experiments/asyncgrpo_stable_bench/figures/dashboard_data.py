"""CSV files for the benchmark dashboard (claude.ai artifact): every run in results/raw plus the training runs still in flight.

    python figures/dashboard_data.py      # in a hopper-cpu job; writes figures/dashboard/*.csv

runs.csv          one row per run: library, workload, model, arm, job, steps done / planned, status, note
robustness.json   the failure ledger (results/robustness/failures.jsonl), every row
w1_evals.csv      AIME24 / AIME25 avg@16 and MATH-500 per checkpoint, every W1 run and both base models
w1_progress.csv   hours since the first optimizer step and seconds per step, every W1 run
w2_steps.csv      seconds per step and train reward, every W2 run
run_configs.json  per run: model, dataset, library and engine versions, GPUs, and the config files that ran
breakdown.csv     tokens per step and mean step time by component (W1-S, W2), from figures/out-dashboard/tables
memory.csv        peak GPU memory of the trainer and engine GPUs per run, same source
episodes.csv      W2: where an episode's time goes (Harbor phases, model calls), seconds per model call, budget hits

Render figures/out-dashboard first (stable-bench-runs/figures-logs/diag_render.py figures/out-dashboard).
"""

import calendar
import csv
import hashlib
import glob
import json
import os
import re
import subprocess
import time

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BENCH, "results", "raw")
RUNS_DIR = "/fsx/amine_dirhoussi/stable-bench-runs"
OUT = os.path.join(BENCH, "figures", "dashboard")
TB_PY = "/fsx/amine_dirhoussi/envs/bench-skyrl/bin/python"  # the dashboard env has no tensorboard
MODEL = {"S": "2B", "M": "9B", "moe": "35B-A3B", "moehard": "35B-A3B", "moesql": "35B-A3B"}
SIZE_WL = {"moe": "MoE", "moehard": "MoE-hard", "moesql": "MoE-SQL"}  # workload suffix when it is not the size letter
TASKS = {"moehard": "TermGrade-hard", "moesql": "Text-to-SQL"}  # W2 task set per size; TermGrade otherwise
PLANNED = {"w1": 200, "w2": 60}
PLANNED_SIZE = {"moesql": 100}  # text-to-SQL follows the miles recipe length

# Training runs not collected into results/raw yet: (run dir, lib, workload, size, arm, job, metrics format)
LIVE = [
    ("verl/verl-w2-M-shared-371895", "verl", "w2", "M", "shared", "371895", "verl"),
    ("prime-rl/prime-rl-w2-moe-shared-371663", "prime-rl", "w2", "moe", "lora32-ep4", "371663", "prime-rl"),
    ("trl-moe-ep/trl-moe-w2-Qwen3.5-35B-A3B-ep4-bfloat16-lora32-local-st8-372251", "trl", "w2", "moe", "lora32-ep4-local-st8", "372251", "trl"),
    ("trl-moe-ep/trl-moe-w2-Qwen3.5-35B-A3B-ep4-bfloat16-lora32-hard-st8-372854", "trl", "w2", "moehard", "lora32-ep4-local-st8", "372854", "trl"),
    ("prime-rl/prime-rl-w2-moehard-shared-372827", "prime-rl", "w2", "moehard", "lora32-ep4", "372827", "prime-rl"),
    ("trl-moe-ep/trl-t2sql-Qwen3.5-35B-A3B-ep4-lora32-main-373132", "trl", "w2", "moesql", "lora32-ep4-st8", "373132", "trl"),
    ("trl-moe-ep/trl-t2sql-Qwen3.5-35B-A3B-ep4-lora32-salt-374011", "trl", "w2", "moesql", "lora32-ep4-st8 cache salt", "374011", "trl"),
    ("prime-rl/prime-rl-t2sql-Qwen3.5-35B-A3B-lora32-373642", "prime-rl", "w2", "moesql", "lora32", "373642", "prime-rl"),
    ("prime-rl/prime-rl-t2sql-Qwen3.5-35B-A3B-lora32-norefill-389085", "prime-rl", "w2", "moesql", "lora32 no refill", "389085", "prime-rl"),
    ("skyrl/skyrl-t2sql-Qwen3.5-35B-A3B-lora32-373788", "skyrl", "w2", "moesql", "lora32 4 GPUs", "373788", "skyrl"),
    ("skyrl/skyrl-w2-M-recommended-374210", "skyrl", "w2", "M", "recommended", "374210", "skyrl"),
    ("skyrl/skyrl-w1-M-recommended-379394", "skyrl", "w1", "M", "recommended", "379394", "skyrl"),
    ("skyrl/skyrl-w2-M-dppo-379410", "skyrl", "w2", "M", "default loss (dppo)", "379410", "skyrl"),
    ("openrlhf/openrlhf-w2-M-shared-379806", "openrlhf", "w2", "M", "shared", "379806", "openrlhf"),
    ("openrlhf/openrlhf-w2-M-offload-379845", "openrlhf", "w2", "M", "shared + adam offload", "379845", "openrlhf"),
    ("skyrl/skyrl-w2-35b-dppo-379439", "skyrl", "w2", "moe", "default loss (dppo) lora32 tp2-ep4", "379439", "skyrl"),
]
# Runs with no results/raw dir: attempts that never trained.
EXTRA = [
    {"run": "verl-w2-M-shared-371788", "lib": "verl", "model": "9B", "arm": "shared", "note": "failed: vLLM worker died (pidfd_getfd); launcher fixed; rerun 371895",
     "workload": "W2-M", "job": "371788", "steps": 0, "planned": 60, "evals": 0, "status": "failed"},
    {"run": "skyrl-w2-M-recommended-c64-377353", "lib": "skyrl", "model": "9B", "arm": "recommended 64 in flight",
     "note": "audit rerun of its recipe at 64 episodes in flight like the other libraries (recipe: 128): 83 of about 109 episodes timed out; no step 1 after 90 min so its watcher stopped it",
     "workload": "W2-M", "job": "377353", "steps": 0, "planned": 60, "evals": 0, "status": "stopped"},
    {"run": "miles-w2-M-shared-372132", "lib": "miles", "model": "9B", "arm": "shared", "note": "failed at start: shipped async agentic flags rejected by its own check",
     "workload": "W2-M", "job": "372132", "steps": 0, "planned": 60, "evals": 0, "status": "failed"},
    {"run": "trl-moe-w2-Qwen3.5-35B-A3B-ep4-bfloat16-lora32-hard-st8-372822", "lib": "trl", "model": "35B-A3B", "arm": "lora32-ep4-local-st8",
     "note": "failed at start: vLLM startup timed out (FlashInfer JIT cache shared by two venvs); rerun 372854",
     "workload": "W2-MoE-hard", "job": "372822", "steps": 0, "planned": 60, "evals": 0, "status": "failed"},
    {"run": "miles-t2sql-373208", "lib": "miles", "model": "35B-A3B", "arm": "as shipped",
     "note": "failed before step 0: routing replay is on and SGLang's flashinfer_trtllm MoE runner records no routing on B200 (miles-022); audit rerun with the triton runner 377633",
     "workload": "W2-MoE-SQL", "job": "373208", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "miles-t2sql-triton-377633", "lib": "miles", "model": "35B-A3B", "arm": "triton MoE runner",
     "note": "audit rerun with miles' shipped --sglang-moe-runner-backend triton: routing crash gone; the step-0 eval died on our scratch dir permissions (ours); rerun 377684",
     "workload": "W2-MoE-SQL", "job": "377633", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "miles-t2sql-triton-377684", "lib": "miles", "model": "35B-A3B", "arm": "triton MoE runner",
     "note": "step-0 eval ran (validation 0.827 like our base eval); the first train step ran out of memory in unfused attention (miles-023)",
     "workload": "W2-MoE-SQL", "job": "377684", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "verl-t2sql-Qwen3.5-35B-A3B-lora32-rec-373603", "lib": "verl", "model": "35B-A3B", "arm": "lora32 merged",
     "note": "failed at the first weight sync: merged LoRA sends packed 3D expert tensors that vLLM can't load (verl-026)",
     "workload": "W2-MoE-SQL", "job": "373603", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "skyrl-t2sql-Qwen3.5-35B-A3B-lora32-373595", "lib": "skyrl", "model": "35B-A3B", "arm": "lora32",
     "note": "failed at start: Transformer Engine cudnn import (fixed with LD_LIBRARY_PATH); rerun 373617",
     "workload": "W2-MoE-SQL", "job": "373595", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "skyrl-t2sql-Qwen3.5-35B-A3B-lora32-373617", "lib": "skyrl", "model": "35B-A3B", "arm": "lora32",
     "note": "failed at start: workers could not connect to Ray; rerun 373719",
     "workload": "W2-MoE-SQL", "job": "373617", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "skyrl-t2sql-Qwen3.5-35B-A3B-lora32-373719", "lib": "skyrl", "model": "35B-A3B", "arm": "lora32 4 GPUs",
     "note": "failed at start: vLLM needed about 10 min to start on a busy node and SkyRL's default health wait is 600 s; rerun 373788 with it at 1800 s",
     "workload": "W2-MoE-SQL", "job": "373719", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "nemo-rl-t2sql-Qwen3.5-35B-A3B-373717", "lib": "nemo-rl", "model": "35B-A3B", "arm": "full FT",
     "note": "failed at the first env step: our env glue read token_ids that NeMo-RL doesn't pass to envs; rerun 373838",
     "workload": "W2-MoE-SQL", "job": "373717", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "nemo-rl-t2sql-Qwen3.5-35B-A3B-373838", "lib": "nemo-rl", "model": "35B-A3B", "arm": "full FT",
     "note": "OOM in the first train step: full fine-tune of the 35B on one node at 29.7k tokens per micro-batch (the recipe is validated on 4 nodes at 9.2k tokens so this one is on us; nemo-rl-001); LoRA retry 373868",
     "workload": "W2-MoE-SQL", "job": "373838", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "nemo-rl-t2sql-Qwen3.5-35B-A3B-373868", "lib": "nemo-rl", "model": "35B-A3B", "arm": "LoRA r32",
     "note": "failed at setup: its FSDP can't shard the LoRA-wrapped linear-attention and shared-expert layers (mixed dtypes) under expert parallel (nemo-rl-002); next the shipped 2-node full fine-tune (373879)",
     "workload": "W2-MoE-SQL", "job": "373868", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "nemo-rl-t2sql-2n-Qwen3.5-35B-A3B-373879", "lib": "nemo-rl", "model": "35B-A3B", "arm": "full FT 2 nodes",
     "note": "our launch config: CPUS_PER_WORKER unset so ray.sub asked for 96 CPUs per node where Slurm grants 92; fixed by setting it; rerun 374000",
     "workload": "W2-MoE-SQL", "job": "373879", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "nemo-rl-t2sql-2n-Qwen3.5-35B-A3B-374000", "lib": "nemo-rl", "model": "35B-A3B", "arm": "full FT 2 nodes",
     "note": "failed at the first vLLM weight load: pidfd_getfd not permitted (the recipe's expandable_segments with CUDA IPC in our container); rerun 374006 with it off",
     "workload": "W2-MoE-SQL", "job": "374000", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
    {"run": "nemo-rl-t2sql-2n-Qwen3.5-35B-A3B-374006", "lib": "nemo-rl", "model": "35B-A3B", "arm": "full FT 2 nodes",
     "note": "cancelled after 4 min before any step: its nodes went to the TRL cache-salt rerun 374011",
     "workload": "W2-MoE-SQL", "job": "374006", "steps": 0, "planned": 100, "evals": 0, "status": "cancelled"},
    {"run": "prime-rl-t2sql-Qwen3.5-35B-A3B-lora32-373584", "lib": "prime-rl", "model": "35B-A3B", "arm": "lora32",
     "note": "stopped after 22 min: every episode failed at scoring because our env port named a method score (shadows verifiers' Task.score); rerun 373642",
     "workload": "W2-MoE-SQL", "job": "373584", "steps": 0, "planned": 100, "evals": 0, "status": "failed"},
]
# Run dirs collected under another run id (an alias dir in stable-bench-runs/<lib>/ that follows <lib>-<wl>-<size>-<tag>-<job>).
ALIAS = {"trl-moe-w2-Qwen3.5-35B-A3B-ep4-bfloat16-lora32-local-371885": "trl-w2-moe-st4-371885"}
# Runs whose notes don't come from meta.json.
NOTE = {
    "skyrl-w1-S-shared-364970": "excluded: our loss mapping collapsed it (skyrl-013)",
    "skyrl-w1-M-shared-366119": "excluded: our loss mapping collapsed it (skyrl-023)",
    "skyrl-w2-M-shared-365144": "excluded: custom loss and generator override; stopped at step 6",
    "skyrl-w2-M-recommended-374210": "collapsed by step 5",
    "openrlhf-w2-M-shared-379806": "failed in its first train step: CUDA out of memory (ZeRO-3; a 30 GiB allocation with 17 GiB free); a first try (379779) stopped at start on a missing HARBOR_URL",
    "skyrl-w2-35b-dppo-379439": "stopped by choice at step 9: TermGrade is too easy at 35B (train reward .58-.89); 35B comparisons stay on text-to-SQL",
    "openrlhf-w2-M-offload-379845": "failed in its first train step: CUDA out of memory again with its Adam state on CPU (a 30.3 GiB allocation with 30.2 GiB free)",
    "trl-w1-M-recommended-366040": "dropped: too slow next to the other libs",
    "trl-w1-M-shared-370844": "cancelled at step 75: node moved to agentic",
    "miles-w2-M-shared-372145": "failed: no sandbox could start (nested Apptainer in its container)",
    "miles-w2-M-shared-372222": "failed: first train step OOM at 65k context (TE unfused attention; CP unsupported with it)",
}


def read_jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()] if os.path.exists(p) else []


def note_for(run_id, tag, meta):
    v = str(meta.get("validity", ""))
    if v.startswith("patched"):
        return "patched lib (excluded)"
    if v.startswith("custom code"):
        return "custom code (excluded)"
    if "smoke" in tag:
        return "smoke"
    return NOTE.get(run_id, "")


def write(name, rows, cols):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")


DATASET = {
    "W1": "DAPO-Math-17k (train). 32 prompts x 8 samples per step, max completion 16k tokens, thinking off. Eval: AIME24 / AIME25 "
    "avg@16 and MATH-500 at steps 50, 100, 150, 200.",
    "W2": "TermGrade terminal tasks: 28 train / 14 held-out. terminus-2 agent, at most 30 model calls and 600 s per episode, "
    "8 tasks x 8 episodes per step. Held-out eval: 14 tasks x 4 episodes per checkpoint.",
    "W2-MoE-hard": "TermGrade-hard: 44 train / 14 held-out tasks from a 152-task TermGrade pool, kept where the 35B base passed 10-75% of 4 tries. "
    "terminus-2 agent, at most 30 model calls and 600 s per episode, 8 tasks x 8 episodes per step. Held-out eval: 14 tasks x 4 (or x 12) episodes.",
    "W2-MoE-SQL": "Text-to-SQL from miles examples/text_to_sql (BIRD-Platinum data): multi-turn SQL tool calls against in-process SQLite, "
    "train 2,033 tasks. Greedy evals at steps 20-100 on validation (398), arcwise (498) and hard (658, overlaps train).",
}
META_FIELDS = ["model", "lib_version", "lib_sha", "inference_engine", "train_backend", "master_dtype", "attention_backend",
               "sandbox_backend", "max_staleness", "gpus_total", "gpus_inference", "gpus_train", "nodes", "gpu_hours", "start_time",
               "end_time", "status", "validity", "notes", "config_file", "config_sha256"]  # fmt: skip
CONFIG_EXT = (".py", ".sh", ".args", ".yaml", ".toml", ".overrides", ".env", ".txt", ".json")
SKIP_FILES = {"freeze.txt", "harbor_cpu.txt", "kernels.json", "trl_sha", "nodes", "job_start", "vllm_ready", "harbor_ready",
              "vllm_final_metrics.txt", "times.txt", "job_times.txt", "tb_scalars.txt"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def resolve_config(meta, run_dir):
    """The config file text that ran: the file as it is now, else the git revision with the recorded sha256, else a copy
    in the run directory with that sha256. Returns (text, where it came from)."""
    cf, want = meta.get("config_file"), meta.get("config_sha256")
    if not cf:
        return None, "no meta.json yet (run not collected): files below are the copies in its run directory"
    path = os.path.join(BENCH, cf)
    if os.path.exists(path) and sha(open(path, "rb").read()) == want:
        return open(path).read(), f"{cf} (current file, sha256 matches)"
    revs = subprocess.run(["git", "log", "--format=%H", "--", cf], cwd=BENCH, capture_output=True, text=True).stdout.split()
    for r in revs:
        b = subprocess.run(["git", "show", f"{r}:./{cf}"], cwd=BENCH, capture_output=True).stdout
        if b and sha(b) == want:
            return b.decode(), f"{cf} at git {r[:9]} (sha256 matches)"
    for f in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if os.path.isfile(f) and os.path.getsize(f) < 200_000 and sha(open(f, "rb").read()) == want:
            return open(f).read(), f"run dir copy {os.path.basename(f)} (sha256 matches)"
    copy = os.path.join(run_dir, "config" + os.path.splitext(cf)[1])
    if os.path.exists(copy):
        return open(copy).read(), f"run dir copy {os.path.basename(copy)} (sha256 differs from meta.json)"
    return None, f"{cf}: no copy with the recorded sha256 found"


def run_files(run_dir, skip_text):
    """Small text config files saved in the run directory (launch env, accelerate config, full args), not the main config."""
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*"))):
        name = os.path.basename(f)
        if (not os.path.isfile(f) or not name.endswith(CONFIG_EXT) or name in SKIP_FILES or name.startswith(("spans", "mem-"))
                or os.path.getsize(f) > 40_000):  # fmt: skip
            continue
        text = open(f, errors="replace").read()
        if text != skip_text:
            out.append({"name": name, "text": text})
    return out


def write_configs(runs):
    """run_configs.json: per run, what it trained (model, data), how (library, engine, backend, GPUs) and the config files."""
    rows = []
    for r in runs:
        run_id, lib = r["run"], r["lib"]
        raw = os.path.join(RAW, run_id)
        meta = json.load(open(os.path.join(raw, "meta.json"))) if os.path.exists(os.path.join(raw, "meta.json")) else {}
        run_dir = os.path.join(RUNS_DIR, lib, run_id)
        if not os.path.isdir(run_dir):
            run_dir = next(iter(glob.glob(os.path.join(RUNS_DIR, "*", run_id))), run_dir)
        text, source = resolve_config(meta, run_dir)
        files = ([{"name": os.path.basename(meta["config_file"]), "text": text}] if text else []) + run_files(run_dir, text)
        row = {"run": run_id, "lib": lib, "workload": r["workload"], "dataset": DATASET.get(r["workload"], DATASET[r["workload"][:2]]), "config_source": source,
               "run_dir": run_dir if os.path.isdir(run_dir) else "", "files": json.dumps(files)}  # fmt: skip
        for k in META_FIELDS:
            v = meta.get(k)
            row[k] = ", ".join(map(str, v)) if isinstance(v, list) else ("" if v is None else str(v))
        if not row["model"]:
            row["model"] = {"2B": "Qwen/Qwen3.5-2B", "9B": "Qwen/Qwen3.5-9B", "35B-A3B": "Qwen/Qwen3.5-35B-A3B"}[r["model"]]
        rows.append(row)
    with open(os.path.join(OUT, "run_configs.json"), "w") as f:
        json.dump(rows, f)
    print(f"wrote run_configs.json: {len(rows)} rows, {os.path.getsize(os.path.join(OUT, 'run_configs.json')) // 1024} KB")


def live_steps(path, fmt):
    d = os.path.join(RUNS_DIR, path)
    if fmt == "trl":
        rows = [m for m in read_jsonl(os.path.join(d, "metrics.jsonl")) if "train_runtime" not in m]  # last row is the run summary
        return [(m["step"], m.get("time"), m.get("perf/step_s"), m.get("reward")) for m in rows]
    if fmt == "skyrl":  # one reward line per step in train.log; step time = gap between those lines
        ts = [(calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")), float(m.group(2))) for line in open(os.path.join(d, "train.log"), errors="replace")
              if (m := re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+.*reward/avg_raw_reward: ([0-9.]+)", line))]  # fmt: skip
        return [(i + 1, t, t - ts[i - 1][0] if i else None, r) for i, (t, r) in enumerate(ts)]
    if fmt == "openrlhf":  # TensorBoard events via collect/openrlhf.py, run under an env that has tensorboard
        code = ("import importlib.util, json, sys; s = importlib.util.spec_from_file_location('c', sys.argv[1]); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "v, t = m.read_tb(sys.argv[2]); print(json.dumps([[k, t[k], v[k].get('timing/step_total'), v[k].get('rollout/reward_mean')] for k in sorted(v)]))")  # fmt: skip
        out = subprocess.run([TB_PY, "-c", code, os.path.join(BENCH, "collect", "openrlhf.py"), os.path.join(d, "tb")], capture_output=True, text=True, check=True).stdout
        return [tuple(r) for r in json.loads(out)]
    if fmt == "verl":
        return [(m["step"], None, m["data"].get("timing_s/step"), m["data"].get("critic/score/mean")) for m in read_jsonl(os.path.join(d, "metrics.jsonl"))]
    by = {}  # prime-rl file monitor: trainer and orchestrator rows of the same step
    for m in read_jsonl(os.path.join(d, "monitors", "file", "metrics.jsonl")):
        if m.get("step") is None:
            continue
        r = by.setdefault(m["step"], [m["step"], None, None, None])
        if m.get("producer") == "trainer":  # several trainer rows per step; one carries time/step
            r[1] = max(r[1] or 0.0, m["time"])
            r[2] = m.get("time/step", r[2])
        elif m.get("train/agg/all/agent/reward/mean") is not None:
            r[3] = m["train/agg/all/agent/reward/mean"]
    return [tuple(r) for _, r in sorted(by.items()) if r[2] is not None]


RUNNING = set(subprocess.run(["squeue", "-h", "-o", "%i", "-t", "RUNNING"], capture_output=True, text=True).stdout.split())


def w2_evals():
    """W2 held-out evals: every eval job dir whose launch.env names EVAL_MODEL and whose probe log finished all its episodes."""
    rows = []
    for env in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "*", "launch.env"))):
        d = os.path.dirname(env)
        kv = dict(line.strip().split("=", 1) for line in open(env) if "=" in line)
        model, log = kv.get("EVAL_MODEL"), os.path.join(d, "probe-ckpt.log")
        split = kv.get("SPLIT", "eval")
        if not model or split not in ("eval", "hard-eval") or not os.path.exists(log):  # probe/train splits aren't held-out evals
            continue
        done = [m for line in open(log, errors="replace") if (m := re.search(r"(\d+)/\1 done, (\d+) unscored, mean reward ([0-9.]+)", line))]
        if not done:
            continue
        n, unscored, mean = int(done[-1].group(1)), int(done[-1].group(2)), float(done[-1].group(3))
        if model.startswith("Qwen/"):
            run, lib, step = "base-" + model.split("/")[1], "base", 0
        else:
            rel = os.path.relpath(model, RUNS_DIR).split("/")
            lib, run = rel[0], rel[1].removesuffix("-aux")
            if rel[1] == "merged":
                run = re.sub(r"-checkpoint-\d+$", "", rel[2])
            lib = "trl" if lib == "trl-moe-ep" else lib
            step = int(re.search(r"(?:checkpoint-|step_)(\d+)", model).group(1))
        run = ALIAS.get(run, run)
        size = "35B-A3B" if re.search(r"35B|moe", model) else "9B"
        task = "TermGrade-hard" if split == "hard-eval" else "TermGrade"
        rows.append({"eval_job": os.path.basename(d), "run": run, "lib": lib, "model": size, "task": task, "step": step, "mean_reward": mean,
                     "episodes": n, "scored": n - unscored})  # fmt: skip
    write("w2_evals.csv", rows, ["eval_job", "run", "lib", "model", "task", "step", "mean_reward", "episodes", "scored"])


def t2sql_evals():
    """Text-to-SQL greedy evals: one summary.json per evaluated model under trl-moe-ep/t2sql-eval/<run>-checkpoint-<step> (or base-<model>)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "trl-moe-ep", "t2sql-eval", "*", "summary.json"))):
        name = os.path.basename(os.path.dirname(p))
        m = re.match(r"(.+)-checkpoint-(\d+)$", name)
        run, step = (m.group(1), int(m.group(2))) if m else (name, 0)
        lib = run.split("-t2sql")[0] if m else "base"
        for split, v in json.load(open(p)).items():
            if split.startswith("eval-"):
                rows.append({"run": run, "lib": lib, "step": step, "split": split.removeprefix("eval-"), "pass_rate": round(v["pass_rate"], 4),
                             "se": round(v["pass_rate_se"], 4), "episodes": v["episodes"]})  # fmt: skip
    write("t2sql_evals.csv", rows, ["run", "lib", "step", "split", "pass_rate", "se", "episodes"])


DIAG = os.path.join(BENCH, "figures", "out-dashboard", "tables")  # charts.py diagnostics rendered there by diag_render.py
DIAG_WL = {"w1-S_math_qwen3.5-2B_dapo-math-17k": "W1-S", "w2-M_agentic_qwen3.5-9B_termgrade": "W2-M",
           "w2-moe_agentic_qwen3.5-35B-A3B_termgrade": "W2-MoE"}  # fmt: skip


def logged(run_id, field, first):
    """Whether the run's steps.jsonl carries `field` (else the chart counts that time in "other")."""
    rows = [s for s in read_jsonl(os.path.join(RAW, run_id, "steps.jsonl")) if s["step"] >= first]
    return int(any(s.get(field) is not None for s in rows))


def diag_tables():
    """breakdown.csv (tokens per step and mean step time by component) and memory.csv (peak GPU memory), from the
    d_step_breakdown / k_w2_tokens / f_peak_memory tables of charts.py diagnostics."""
    bd, mem = [], []
    for folder, wl in DIAG_WL.items():
        d = os.path.join(DIAG, folder)
        w2 = wl.startswith("W2")
        p = os.path.join(d, "k_w2_tokens.csv" if w2 else "d_step_breakdown.csv")
        for r in csv.DictReader(open(p)) if os.path.exists(p) else []:
            trained = float(r["big"]) / 1e6
            bd.append({"workload": wl, "run": r["run_id"], "lib": r["lib"], "steps": r["steps"], "trained_M": round(trained, 4),
                       "useful_M": round(trained if w2 else float(r["useful_M"]), 4),
                       "fwd_M": round(float(r["small"] if w2 else r["fwd"]) / 1e6, 4),
                       "useful_exact": int(w2 or r["useful_exact"] == "True"),
                       **{k: round(float(r[k]), 3) for k in ("train_compute", "wait_for_rollouts", "weight_sync", "other")},
                       "wait_logged": logged(r["run_id"], "wait_for_rollouts_time", 2 if w2 else 6),
                       "sync_logged": logged(r["run_id"], "weight_sync_time", 2 if w2 else 6)})  # fmt: skip
        p = os.path.join(d, "f_peak_memory.csv")
        for r in csv.DictReader(open(p)) if os.path.exists(p) else []:
            num = lambda k: round(float(r[k]), 1) if r[k] else ""  # noqa: E731
            mem.append({"workload": wl, "run": r["run_id"], "lib": r["lib"], "train_peak_gb": num("train_peak_gb"),
                        "engine_peak_gb": num("engine_peak_gb"), "torch_peak_gb": num("torch_peak_gb"),
                        "train_gpus": len(r["train_gpus"].split()), "engine_gpus": len(r["engine_gpus"].split())})  # fmt: skip
    write("breakdown.csv", bd, ["workload", "run", "lib", "steps", "trained_M", "useful_M", "fwd_M", "useful_exact", "train_compute",
                                "wait_for_rollouts", "weight_sync", "other", "wait_logged", "sync_logged"])  # fmt: skip
    write("memory.csv", mem, ["workload", "run", "lib", "train_peak_gb", "engine_peak_gb", "torch_peak_gb", "train_gpus", "engine_gpus"])


def episodes_table():
    """episodes.csv: per W2 run, where an episode's wall time goes (means over its finished episodes, so the parts add
    up): Harbor's environment_setup, agent_setup, model calls (model_s), the rest of agent_execution (commands in the
    sandbox, terminus-2 waits and Harbor's own overhead), verifier, and what Harbor's phases don't cover."""
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, "*-w2-*", ""))):
        run_id = os.path.basename(d.rstrip("/"))
        if "smoke" in run_id:
            continue
        E = [e for e in read_jsonl(d + "episodes.jsonl") if e.get("wall_s") is not None and e.get("model_s") is not None]
        if not E:
            continue
        ph = lambda e, k: (e.get("phases") or {}).get(k) or 0.0  # noqa: E731
        mean = lambda f: sum(f(e) for e in E) / len(E)  # noqa: E731
        walls = sorted(e["wall_s"] for e in E)
        exec_ = mean(lambda e: ph(e, "agent_execution"))
        model = mean(lambda e: e["model_s"])
        parts = {"env_setup": mean(lambda e: ph(e, "environment_setup")), "agent_setup": mean(lambda e: ph(e, "agent_setup")),
                 "model": min(model, exec_) if exec_ else model, "commands": max(0.0, exec_ - model),
                 "verifier": mean(lambda e: ph(e, "verifier"))}  # fmt: skip
        wall = mean(lambda e: e["wall_s"])
        parts["other"] = max(0.0, wall - sum(parts.values()))
        turns = sum(e.get("turns") or 0 for e in E)
        out.append({"workload": "W2-MoE" if "-w2-moe-" in run_id else "W2-M", "run": run_id, "lib": run_id.split("-w2-")[0],
                    "episodes": len(E), "wall_mean": round(wall, 1), "wall_p50": round(walls[len(walls) // 2], 1),
                    "wall_p90": round(walls[int(len(walls) * 0.9)], 1), **{k: round(v, 1) for k, v in parts.items()},
                    "turns_mean": round(turns / len(E), 1), "s_per_call": round(sum(e["model_s"] for e in E) / turns, 2) if turns else "",
                    "budget_frac": round(sum(e.get("status") == "budget_scored" for e in E) / len(E), 3),
                    "failed_frac": round(sum(e.get("status") == "failed" for e in E) / len(E), 3)})  # fmt: skip
    write("episodes.csv", out, ["workload", "run", "lib", "episodes", "wall_mean", "wall_p50", "wall_p90", "env_setup", "agent_setup", "model",
                                "commands", "verifier", "other", "turns_mean", "s_per_call", "budget_frac", "failed_frac"])  # fmt: skip


def robustness():
    """robustness.json: every row of the hand-kept failure ledger (results/robustness/failures.jsonl); the page counts kind=failure."""
    keep = ["id", "date", "lib", "phase", "category", "severity", "kind", "attributable_to", "symptom", "root_cause", "fix",
            "patched_lib_source", "time_lost_min", "source"]  # fmt: skip
    rows = [{k: r.get(k) for k in keep} for r in read_jsonl(os.path.join(BENCH, "results", "robustness", "failures.jsonl"))]
    with open(os.path.join(OUT, "robustness.json"), "w") as f:
        json.dump(rows, f)
    print(f"wrote robustness.json: {len(rows)} rows")


def main():
    os.makedirs(OUT, exist_ok=True)
    robustness()
    w2_evals()
    t2sql_evals()
    diag_tables()
    episodes_table()
    runs, evals, progress, w2 = [], [], [], []
    for d in sorted(glob.glob(os.path.join(RAW, "*", ""))):
        run_id = os.path.basename(d.rstrip("/"))
        meta = json.load(open(d + "meta.json")) if os.path.exists(d + "meta.json") else {}
        ev = read_jsonl(d + "eval.jsonl")
        if run_id.startswith("base-"):
            for e in ev:
                evals.append({"run": run_id, "lib": "base", "model": run_id.split("Qwen3.5-")[1], "arm": "base", "note": "",
                              "step": 0, "aime24": e.get("aime24_avg16"), "aime25": e.get("aime25_avg16"), "math500": e.get("math500_avg1")})  # fmt: skip
            continue
        lib = meta.get("lib") or run_id.split("-w")[0]
        rest = run_id[len(lib) + 1 :].split("-")  # <workload>-<size>-<tag...>-<job id>
        wl, size, tag, job = rest[0], rest[1], "-".join(rest[2:-1]), rest[-1]
        note = note_for(run_id, tag, meta)
        steps = sorted(read_jsonl(d + "steps.jsonl"), key=lambda s: s["step"])
        base = {"run": run_id, "lib": lib, "model": MODEL[size], "arm": tag, "note": note}
        done = int(steps[-1]["step"]) if steps else 0
        status = "smoke" if "smoke" in tag else "finished" if done >= PLANNED[wl] else "running" if job in RUNNING else "stopped"
        if meta.get("status") == "failed" and status == "stopped":
            status = "failed"
        if status == "running":
            base["note"] = "running"
        runs.append({**base, "workload": wl.upper() + "-" + SIZE_WL.get(size, size), "job": job, "steps": done, "planned": PLANNED[wl],
                     "evals": len(ev), "status": status})  # fmt: skip
        for e in ev:
            evals.append({**base, "step": e["step"], "aime24": e.get("aime24_avg16"), "aime25": e.get("aime25_avg16"),
                          "math500": e.get("math500_avg1")})  # fmt: skip
        t0 = steps[0]["time"] if steps else None
        for s in steps:
            row = {**base, "step": s["step"], "hours": round((s["time"] - t0) / 3600, 3), "step_time_s": s.get("step_time"),
                   "train_reward": s.get("reward_mean")}  # fmt: skip
            if wl == "w1":
                progress.append(row)
            else:
                w2.append({**row, "task": TASKS.get(size, "TermGrade")})
    for path, lib, wl, size, tag, job, fmt in LIVE:
        run_id = os.path.basename(path)
        if os.path.isdir(os.path.join(RAW, run_id)):  # collected since: results/raw has it
            continue
        m = live_steps(path, fmt)
        ev = read_jsonl(os.path.join(RUNS_DIR, path, "eval.jsonl"))  # in-job W1 evals, until results/raw collects the run
        planned = PLANNED_SIZE.get(size, PLANNED[wl])
        status = "running" if job in RUNNING else "finished" if m and m[-1][0] >= planned else "stopped"
        base = {"run": run_id, "lib": lib, "model": MODEL[size], "arm": tag, "note": "running" if status == "running" else NOTE.get(run_id, "")}
        runs.append({**base, "workload": wl.upper() + "-" + SIZE_WL.get(size, size), "job": job, "steps": m[-1][0] if m else 0,
                     "planned": planned, "evals": len(ev), "status": status})  # fmt: skip
        for e in ev:
            evals.append({**base, "step": e["step"], "aime24": e.get("aime24_avg16"), "aime25": e.get("aime25_avg16"),
                          "math500": e.get("math500_avg1")})  # fmt: skip
        t0 = m[0][1] if m else None
        for step, t, step_s, reward in m:
            row = {**base, "step": step, "hours": round((t - t0) / 3600, 3) if t and t0 else "", "step_time_s": step_s, "train_reward": reward}
            if wl == "w1":
                progress.append(row)
            else:
                w2.append({**row, "task": TASKS.get(size, "TermGrade")})
    runs += [r for r in EXTRA if not os.path.isdir(os.path.join(RAW, r["run"]))]
    write_configs(runs)
    cols = ["run", "lib", "model", "arm", "note"]
    write("runs.csv", runs, cols + ["workload", "job", "steps", "planned", "evals", "status"])
    write("w1_evals.csv", evals, cols + ["step", "aime24", "aime25", "math500"])
    write("w1_progress.csv", progress, cols + ["step", "hours", "step_time_s", "train_reward"])
    write("w2_steps.csv", w2, cols + ["task", "step", "hours", "step_time_s", "train_reward"])


if __name__ == "__main__":
    main()
