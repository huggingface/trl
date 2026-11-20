"""In-job watcher for launch/nemo-rl/w2/w2.sbatch. Polls the run's tensorboard every 60 s and kills the driver's
process group (`WATCH_KILL <reason>` in watch.log) on:
- a CUDA OOM in train.log;
- no step 1 within 90 min of the driver start (FIRST_TIMEOUT), or no new step for 90 min later;
- any step: non-finite loss; steps 1-3: token_mult_prob_error > 1.1 or gen_kl_error > 0.05;
- collapse: 3 consecutive steps with mean reward 0 after the run's first 5 steps averaged > 0.2.
Logs one line per step (loss, reward, step time, memory) to watch.log.

    python watch.py <run_dir> <driver_pid>
"""

import glob
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
    open(os.path.join(run_dir, "WATCH_KILL"), "w").write(reason + "\n")
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


def n_loss_events():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    n = 0
    for f in glob.glob(os.path.join(run_dir, "logs", "**", "events.out.tfevents*"), recursive=True):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        n += sum(len(acc.Scalars(t)) for t in acc.Tags()["scalars"] if t.endswith("train/loss"))
    return n


def find(metrics, *names):
    for k, v in metrics.items():
        if any(k.endswith(n) for n in names):
            return v
    return None


t0, last_step, last_change, checked, rewards = time.time(), None, time.time(), set(), {}
last_step0 = None
while True:
    time.sleep(60)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        say("driver exited")
        break
    train_log = open(os.path.join(run_dir, "train.log"), errors="ignore").read()[-300000:]
    if "CUDA out of memory" in train_log or "OutOfMemoryError" in train_log:
        kill("CUDA OOM in train.log")
    try:
        steps = scalars()
    except Exception as exc:  # event file mid-write
        say(f"tensorboard read failed: {exc!r}")
        continue
    done = sorted(s for s, m in steps.items() if find(m, "train/loss") is not None)
    n_loss = n_loss_events()
    if last_step0 is None:
        last_step0 = n_loss  # loss events already there at start (resume): the first-step clock waits for a new one
    for s in done:
        if s in checked:
            continue
        checked.add(s)
        m = steps[s]
        loss, perr, kl = find(m, "train/loss"), find(m, "train/token_mult_prob_error"), find(m, "train/gen_kl_error")
        reward = find(m, "train/reward", "train/avg_reward", "train/total_reward")
        rewards[s] = reward
        mem = {k.split("/")[-1]: round(v, 1) for k, v in m.items() if "memory" in k and "peak" in k}
        say(f"step {s}: loss {loss} reward {reward} prob_err {perr} gen_kl {kl} "
            f"step time {find(m, 'timing/train/total_step_time')} mem {mem}")
        if loss is None or not math.isfinite(loss):
            kill(f"step {s}: loss {loss}")
        if len(checked) <= 3:
            if perr is not None and perr > 1.1:
                kill(f"step {s}: token_mult_prob_error {perr:.3f} > 1.1")
            if kl is not None and kl > 0.05:
                kill(f"step {s}: gen_kl_error {kl:.4f} > 0.05")
        first = [rewards[k] for k in sorted(rewards)[:5] if rewards[k] is not None]
        last3 = [rewards[k] for k in sorted(rewards)[-3:]]
        if len(rewards) >= 8 and len(first) == 5 and sum(first) / 5 > 0.2 and all(r == 0 for r in last3):
            kill(f"collapse: reward 0 on steps {sorted(rewards)[-3:]}")
    # progress = a new train/loss event in any event file (a resumed run logs step numbers the first attempt already logged)
    if n_loss != last_step:
        last_step, last_change = n_loss, time.time()
    if last_step == last_step0 and time.time() - t0 > FIRST_TIMEOUT:
        kill(f"no first step after {FIRST_TIMEOUT:.0f} s")
    if n_loss and time.time() - last_change > 90 * 60:
        kill(f"no new step for 90 min (last {last_step})")
