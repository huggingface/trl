"""In-job watcher for SkyRL's recommended W2 run: one line per step in RUN_DIR/watch.log, scancel on failure.

Every 60 s reads RUN_DIR/tb, gpu.csv, train.log and the Harbor trials dir. Kills the job when:
  - no step 1 within FIRST_STEP_MIN (90) minutes of job start, or no new step for STALL_MIN (180) minutes after one;
  - OOM in train.log (CUDA out of memory / OutOfMemoryError);
  - a non-finite loss or grad norm;
  - all of the first 32+ finished trials (or the last 64) ended with an exception;
  - collapse: reward <= 0.02 and grad norm <= 1e-4 for 3 consecutive steps (added 2026-11-14 for the audit reruns).
Usage: python w2rec_watch.py <run dir> <trials dir>
"""

import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN, TRIALS = sys.argv[1], sys.argv[2]
FIRST_STEP_MIN = float(os.environ.get("FIRST_STEP_MIN", 90))
STALL_MIN = float(os.environ.get("STALL_MIN", 180))
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
    if not os.path.isdir(f"{RUN}/tb"):
        return {}
    acc = EventAccumulator(f"{RUN}/tb", size_guidance={"scalars": 0})
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
                    peak[int(row[1])] = max(peak.get(int(row[1]), 0), int(row[3].split()[0]))
    except (OSError, ValueError, IndexError):
        pass
    return peak


def trials():
    rows = []
    for p in glob.glob(f"{TRIALS}/**/result.json", recursive=True):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        if r.get("finished_at") is None:
            continue
        exc = (r.get("exception_info") or {}).get("exception_type")
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rows.append((r.get("finished_at"), exc, rew))
    return sorted(rows, key=lambda x: str(x[0]))


OOM = re.compile(r"CUDA out of memory|OutOfMemoryError")
seen, last_new, log_pos, flat = set(), None, 0, 0
log(f"watching {RUN}")
while True:
    time.sleep(60)
    try:
        with open(f"{RUN}/train.log", errors="replace") as f:
            f.seek(log_pos)
            chunk = f.read()
            log_pos = f.tell()
        if OOM.search(chunk):
            kill("OOM in train.log: " + OOM.search(chunk).group(0))
    except OSError:
        pass
    tr = trials()
    errs = [t for t in tr if t[1] is not None]
    window = tr[-64:]
    if len(tr) >= 32 and (len(errs) == len(tr) or (len(window) == 64 and all(t[1] is not None for t in window))):
        kill(f"all of {len(tr) if len(errs) == len(tr) else 64} finished trials errored; e.g. {errs[-1][1]}")
    steps = scalars()
    for s in sorted(s for s in steps if s not in seen):
        m = steps[s]
        if not seen:
            log("tags: " + " ".join(sorted(m)))
        loss = find(m, "policy_loss")
        grad = find(m, "grad_norm")
        peak = gpu_peak()
        exc = {}
        for t in tr:
            exc[t[1]] = exc.get(t[1], 0) + 1
        log(
            f"step {s}: step_time {find(m, 'timing/step')} gen_wait {find(m, 'wait_for_generation')} train {find(m, 'timing/train')} "
            f"sync {find(m, 'sync_weights')} reward {find(m, 'avg_raw_reward')} loss {loss} grad_norm {grad} "
            f"logp_gap {find(m, 'logprobs_abs_diff_mean')} | trials {len(tr)} exceptions {exc} "
            f"| gpu peak MiB {dict(sorted(peak.items()))}"
        )
        seen.add(s)
        last_new = time.time()
        for name, v in (("loss", loss), ("grad norm", grad)):
            if v is not None and not math.isfinite(v):
                kill(f"step {s}: {name} {v}")
        rew = find(m, "avg_raw_reward")
        flat = flat + 1 if (rew is not None and grad is not None and rew <= 0.02 and grad <= 1e-4) else 0
        if flat >= 3:
            kill(f"collapse: reward ~0 and grad norm ~0 for 3 steps (last step {s})")
    if not seen and time.time() - T0 > FIRST_STEP_MIN * 60:
        kill(f"no step 1 after {FIRST_STEP_MIN} min")
    if last_new and time.time() - last_new > STALL_MIN * 60:
        kill(f"no new step for {STALL_MIN} min")
