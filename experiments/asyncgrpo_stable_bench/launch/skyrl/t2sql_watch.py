"""In-job watcher for the SkyRL text-to-SQL run: logs the first steps and cancels the job on failure.

Every 60 s reads the tensorboard scalars, gpu.csv and episodes.jsonl of RUN_DIR and appends a line per new step to
RUN_DIR/watch.log (memory, step time, logprob gap, reward, episodes). Cancels the job (scancel) when:
  - no step 1 within FIRST_STEP_MIN minutes of job start, or no new step for STALL_MIN minutes after one;
  - mean |logp_train - logp_vllm| > 0.25 (a packing / weight-sync bug, base TRL gap is ~0.017);
  - grad norm NaN/inf;
  - from step 6, raw reward < 0.2 three steps in a row (base pass rate ~0.77).
Usage: python t2sql_watch.py <run dir>
"""

import csv
import glob
import json
import math
import os
import subprocess
import sys
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = sys.argv[1]
FIRST_STEP_MIN = float(os.environ.get("FIRST_STEP_MIN", 150))
STALL_MIN = float(os.environ.get("STALL_MIN", 75))
T0 = time.time()


def log(msg):
    with open(f"{RUN}/watch.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def kill(reason):
    log(f"KILL: {reason}")
    open(f"{RUN}/WATCH_KILLED", "w").write(reason + "\n")
    subprocess.run(["scancel", os.environ["SLURM_JOB_ID"]])
    sys.exit(1)


def find(m, *subs):
    for k, v in m.items():
        if all(s in k for s in subs):
            return v
    return None


def scalars():
    tb = f"{RUN}/tb"
    if not os.path.isdir(tb):
        return {}
    acc = EventAccumulator(tb, size_guidance={"scalars": 0})
    acc.Reload()
    steps = {}
    for tag in acc.Tags()["scalars"]:
        for ev in acc.Scalars(tag):
            steps.setdefault(ev.step, {})[tag] = ev.value
    return steps


def gpu_peak():
    peak = {}
    try:
        with open(f"{RUN}/gpu.csv") as f:
            for row in csv.reader(f):
                if len(row) > 3 and row[1].strip().isdigit():
                    peak[row[1].strip()] = max(peak.get(row[1].strip(), 0), int(row[3].split()[0]))
    except (OSError, ValueError, IndexError):
        pass
    return peak


def episodes():
    n, correct, score = 0, 0.0, 0.0
    try:
        with open(f"{RUN}/episodes.jsonl") as f:
            for line in f:
                r = json.loads(line)
                n += 1
                correct += r["correct"]
                score += r["score"]
    except (OSError, ValueError, KeyError):
        pass
    return n, correct / max(n, 1), score / max(n, 1)


seen, low, last_new = set(), 0, None
log(f"watching {RUN}")
while True:
    time.sleep(60)
    # a crashed vLLM TP rank leaves its peers waiting forever (skyrl-026): its error only shows in the Ray worker logs
    for f in glob.glob(f"{RUN}/ray_logs/worker-*.out"):
        hit = subprocess.run(["grep", "-m1", "-E", "InductorError|ERROR .*multiproc_executor", f], capture_output=True, text=True).stdout.strip()
        if hit:
            kill(f"worker crash in {os.path.basename(f)}: {hit[-200:]}")
    steps = scalars()
    new = sorted(s for s in steps if s not in seen)
    for s in new:
        m = steps[s]
        if s == min(steps) and not seen:
            log("tags: " + " ".join(sorted(m)))
        gap = find(m, "logprobs_abs_diff_mean")
        grad = find(m, "grad_norm")
        rew = find(m, "avg_raw_reward")
        n, corr, sc = episodes()
        peak = gpu_peak()
        log(
            f"step {s}: step_time {find(m, 'timing/step')} gen {find(m, 'timing/generate')} train {find(m, 'timing/train')} "
            f"reward {rew} logp_gap {gap} grad_norm {grad} lr {find(m, 'policy_lr')} | episodes {n} correct {corr:.3f} "
            f"score {sc:.3f} | gpu peak MiB {max(peak.values()) if peak else None}"
        )
        seen.add(s)
        last_new = time.time()
        if gap is not None and gap > 0.25:
            kill(f"step {s}: logprob gap {gap:.3f}")
        if grad is not None and not math.isfinite(grad):
            kill(f"step {s}: grad norm {grad}")
        if rew is not None and s >= 6:
            low = low + 1 if rew < 0.2 else 0
            if low >= 3:
                kill(f"step {s}: reward < 0.2 for 3 steps")
    if not seen and time.time() - T0 > FIRST_STEP_MIN * 60:
        kill(f"no step 1 after {FIRST_STEP_MIN} min")
    if last_new and time.time() - last_new > STALL_MIN * 60:
        kill(f"no new step for {STALL_MIN} min")
