"""In-job watcher for the prime-rl text-to-SQL run (launch/prime-rl/w2_t2sql.sbatch): reads the run's file monitor
(monitors/file/metrics.jsonl) and logs each of the first STEPS steps; cancels the job if the run is clearly broken.

Kills on: no step 1 within FIRST_STEP_S; a step slower than STEP_S; trainer peak memory > 170 GiB; mismatch KL > 0.05;
more than 20% of episodes with an error; no completed episodes; reward outside [-1.2, 1.1]. Writes OK to <aux>/WATCH_OK
after STEPS healthy steps and exits.

Usage: python t2sql_watch.py <run dir> <aux dir> <job id>
"""

import json
import os
import subprocess
import sys
import time

run, aux, job = sys.argv[1:4]
STEPS = int(os.environ.get("WATCH_STEPS", "5"))
FIRST_STEP_S = float(os.environ.get("WATCH_FIRST_STEP_S", str(80 * 60)))
STEP_S = float(os.environ.get("WATCH_STEP_S", str(40 * 60)))
log = open(os.path.join(aux, "watch.log"), "a", buffering=1)


def say(msg):
    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def kill(reason):
    say(f"KILL: {reason}")
    open(os.path.join(aux, "WATCH_KILLED"), "w").write(reason + "\n")
    subprocess.run(["scancel", job])
    sys.exit(1)


def rows():
    path = os.path.join(run, "monitors/file/metrics.jsonl")
    if not os.path.exists(path):
        return {}, {}
    trainer, orch = {}, {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line being written
        if "step" not in r:
            continue
        (trainer if r.get("producer") == "trainer" else orch).setdefault(r["step"], {}).update(r)
    return trainer, orch


def pick(r, suffix):
    keys = [k for k in r if k.endswith(suffix) and "/all/" in k and "/effective/" not in k and k.startswith("train/t2sql")]
    return r[keys[0]] if keys else None


t0 = time.time()
last_t, done = t0, 0
say(f"watching {run}, {STEPS} steps")
while done < STEPS:
    time.sleep(30)
    trainer, orch = rows()
    ready = sorted(s for s in trainer if s in orch and s > done)
    if not ready:
        limit = FIRST_STEP_S if done == 0 else STEP_S
        if time.time() - last_t > limit:
            kill(f"no step {done + 1} after {time.time() - last_t:.0f} s")
        continue
    for s in ready:
        t, o = trainer[s], orch[s]
        mem = t.get("perf/peak_memory")
        kl = t.get("mismatch_kl/all/mean")
        reward = pick(o, "/agent/reward/mean")
        err = pick(o, "/has_error/mean")
        completed = pick(o, "/agent/is_completed/mean")
        branches = pick(o, "/num_branches/mean")
        say(f"step {s}: time/step {t.get('time/step', 0):.0f}s fwd_bwd {t.get('time/forward_backward', 0):.0f}s "
            f"wait {t.get('time/wait_for_batch', 0):.0f}s peak_mem {mem} mismatch_kl {kl} grad_norm {t.get('optim/grad_norm')} "
            f"lr {t.get('optim/lr')} | reward {reward} completed {completed} error {err} branches {branches} "
            f"turns {pick(o, '/num_turns/mean')} out_tok {pick(o, '/num_output_tokens/mean')} "
            f"rollouts {o.get('progress/rollouts')} staleness_max {o.get('staleness/rollout/max')}")
        if mem is not None and mem > 170:
            kill(f"step {s}: peak memory {mem} GiB")
        if kl is not None and kl > 0.05:
            kill(f"step {s}: mismatch KL {kl}")
        if err is not None and err > 0.2:
            kill(f"step {s}: {err:.0%} of episodes errored")
        if completed is not None and completed == 0:
            kill(f"step {s}: no completed episodes")
        if reward is not None and not -1.2 <= reward <= 1.1:
            kill(f"step {s}: reward {reward}")
        if s > 1 and t.get("time/step", 0) > STEP_S:
            kill(f"step {s}: {t['time/step']:.0f} s")
        done = s
        last_t = time.time()
say(f"first {STEPS} steps OK ({time.time() - t0:.0f} s since watch start)")
open(os.path.join(aux, "WATCH_OK"), "w").write(f"{done}\n")
