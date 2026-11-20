"""In-job watcher for launch/nemo-rl/w1/w1.sbatch. Runs inside the container next to the driver,
polls the run dir every 60 s, and kills the driver's process group with `WATCH_KILL <reason>` in watch.log on:
- a CUDA OOM / Traceback-with-exit in train.log;
- no first step within FIRST_TIMEOUT s, or no new step for 90 min later;
- steps 1-3: non-finite loss, train/token_mult_prob_error > 1.1, or train/gen_kl_error > 0.05;
- collapse: from step 10 on, mean reward of the last 5 steps below 0.05 (base Qwen3.5-2B starts near 0.4).
It also logs a one-line summary per step (reward, step time, memory) to watch.log.

    python watch.py <run_dir> <driver_pid>
"""

import glob
import json
import math
import os
import signal
import sys
import time


run_dir, pid = sys.argv[1], int(sys.argv[2])
FIRST_TIMEOUT = float(os.environ.get("FIRST_TIMEOUT", 90 * 60))
log = open(os.path.join(run_dir, "watch.log"), "a", buffering=1)


def say(msg):
    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def kill(reason):
    say(f"WATCH_KILL {reason}")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            break
        time.sleep(30)
    sys.exit(1)


def scalars():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for f in glob.glob(os.path.join(run_dir, "logs", "**", "events.out.tfevents*"), recursive=True):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for e in acc.Scalars(tag):
                out.setdefault(e.step, {})[tag] = e.value
    return out


def find(metrics, *names):
    for k, v in metrics.items():
        if any(k.endswith(n) for n in names):
            return v
    return None


t0, last_step, last_change, checked, episodes_checked = time.time(), 0, time.time(), set(), False
while True:
    time.sleep(60)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        say("driver exited")
        break
    train_log = open(os.path.join(run_dir, "train.log"), errors="ignore").read()[-200000:]
    if "CUDA out of memory" in train_log or "OutOfMemoryError" in train_log:
        kill("CUDA OOM in train.log")

    try:
        steps = scalars()
    except Exception as exc:  # event file mid-write
        say(f"tensorboard read failed: {exc!r}")
        continue
    done = sorted(s for s, m in steps.items() if find(m, "train/loss") is not None)
    for s in done:
        if s in checked:
            continue
        checked.add(s)
        m = steps[s]
        loss, perr, kl = find(m, "train/loss"), find(m, "train/token_mult_prob_error"), find(m, "train/gen_kl_error")
        mem = {k.split("/")[-1]: round(v, 1) for k, v in m.items() if "memory" in k and "peak" in k}
        say(f"step {s}: loss {loss} reward {find(m, 'train/reward', 'train/avg_reward')} prob_err {perr} gen_kl {kl} "
            f"step time {find(m, 'timing/train/total_step_time')} mem {mem}")
        if len(checked) <= 3:
            if loss is None or not math.isfinite(loss):
                kill(f"step {s}: loss {loss}")
            if perr is not None and perr > 1.1:
                kill(f"step {s}: token_mult_prob_error {perr:.3f} > 1.1")
            if kl is not None and kl > 0.05:
                kill(f"step {s}: gen_kl_error {kl:.4f} > 0.05")
    rew = [find(steps[s], "train/reward", "train/avg_reward") for s in done[-5:]]
    if len(done) >= 10 and all(r is not None for r in rew) and sum(rew) / len(rew) < 0.05:
        kill(f"collapse: mean reward of steps {done[-5:]} is {sum(rew) / len(rew):.3f}")
    if done and done[-1] != last_step:
        last_step, last_change = done[-1], time.time()
    if not done and time.time() - t0 > FIRST_TIMEOUT:
        kill(f"no first step after {FIRST_TIMEOUT:.0f} s")
    if done and time.time() - last_change > 90 * 60:
        kill(f"no new step for 90 min (last {last_step})")
