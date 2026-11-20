"""In-job watcher for SkyRL W1 runs: one line per step in RUN_DIR/watch.log, scancel on failure.

Every 60 s reads RUN_DIR/tb and train.log. Kills the job when:
  - no step 1 within FIRST_STEP_MIN (90) minutes of job start, or no new step for STALL_MIN (60) minutes after one;
  - OOM in train.log;
  - a non-finite loss or grad norm;
  - collapse: reward <= 0.02 and grad norm <= 1e-4 for 3 consecutive steps.
Usage: python w1_watch.py <run dir>
"""

import math
import os
import re
import subprocess
import sys
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = sys.argv[1]
FIRST_STEP_MIN = float(os.environ.get("FIRST_STEP_MIN", 90))
STALL_MIN = float(os.environ.get("STALL_MIN", 60))
T0 = time.time()
OOM = re.compile(r"CUDA out of memory|OutOfMemoryError")


def log(msg):
    with open(f"{RUN}/watch.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def kill(reason):
    log(f"KILL: {reason}")
    open(f"{RUN}/WATCH_KILLED", "w").write(reason + "\n")
    subprocess.run(["scancel", os.environ["SLURM_JOB_ID"]])
    sys.exit(1)


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
    steps = scalars()
    for s in sorted(s for s in steps if s not in seen):
        m = steps[s]
        rew, grad, loss = m.get("reward/avg_raw_reward"), m.get("policy/grad_norm"), m.get("policy/policy_loss")
        log(f"step {s}: step_time {m.get('timing/step')} reward {rew} loss {loss} grad_norm {grad} "
            f"resp_len {m.get('generate/avg_num_tokens')} staleness_max {m.get('async/staleness_max')}")
        seen.add(s)
        last_new = time.time()
        for name, v in (("loss", loss), ("grad norm", grad)):
            if v is not None and not math.isfinite(v):
                kill(f"step {s}: {name} {v}")
        flat = flat + 1 if (rew is not None and grad is not None and rew <= 0.02 and grad <= 1e-4) else 0
        if flat >= 3:
            kill(f"collapse: reward ~0 and grad norm ~0 for 3 steps (last step {s})")
    if not seen and time.time() - T0 > FIRST_STEP_MIN * 60:
        kill(f"no step 1 after {FIRST_STEP_MIN} min")
    if last_new and time.time() - last_new > STALL_MIN * 60:
        kill(f"no new step for {STALL_MIN} min")
