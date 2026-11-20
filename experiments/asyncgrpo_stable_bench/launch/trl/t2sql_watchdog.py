"""
In-job checks for `launch/trl/t2sql_train.sbatch` (no separate smoke job), `launch/trl/w2_watchdog.py` without the Harbor parts:
watches the run dir while the trainer runs and stops the trainer with a clear message on the first failure. Writes every
verdict to <run_dir>/watchdog.txt.

    python t2sql_watchdog.py <run_dir> <check_steps> <first_step_timeout_s>

1. First `check_steps` steps: each must log a finite loss, a weight sync under 60 s, a trainer-vs-vLLM logp gap
   (`sampling/sampling_logp_difference/mean`) under 0.05, and peak reserved memory under 175 GB on every rank. Step 1
   must arrive within `first_step_timeout_s`.
2. Episodes: the first 64 finished episodes (episodes.jsonl) are summarized (reward, correct, turns, status); fewer than
   32 with a sampled token, or all of them unanswered, stops the run.
3. Checkpoints: each `checkpoint-N` is merged into a full HF model (`merge_lora.py`, CPU) once the next step has
   logged. The first merge must succeed with nonzero LoRA weights; later merge failures are reported only. Merged
   models land in stable-bench-runs/trl-moe-ep/merged/<run_id>-checkpoint-N.
"""

import glob
import json
import math
import os
import subprocess
import sys
import time


run_dir, check_steps, first_timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
run_id = os.path.basename(run_dir)
here = os.path.dirname(os.path.abspath(__file__))
merged_root = "/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/merged"
log = open(os.path.join(run_dir, "watchdog.txt"), "a", buffering=1)


def say(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    log.write(line + "\n")
    print(f"[watchdog] {line}", flush=True)


def fail(msg):
    say(f"WATCHDOG FAIL: {msg}. Stopping the trainer.")
    open(os.path.join(run_dir, "WATCHDOG_FAILED"), "w").write(msg + "\n")
    subprocess.run(["pkill", "-TERM", "-u", os.environ["USER"], "-f", "t2sql_train.py"])
    time.sleep(60)
    subprocess.run(["pkill", "-KILL", "-u", os.environ["USER"], "-f", "t2sql_train.py"])
    sys.exit(1)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written
            pass
    return rows


start = time.time()
checked_steps, episodes_checked, merged = set(), False, {}
while True:
    steps = [r for r in read_jsonl(os.path.join(run_dir, "metrics.jsonl")) if "loss" in r]
    if not steps and time.time() - start > first_timeout:
        fail(f"no training step logged {first_timeout:.0f} s after the trainer started")

    for r in steps:
        s = r["step"]
        if s > check_steps or s in checked_steps:
            continue
        mem = [m for rank in range(8) for m in read_jsonl(os.path.join(run_dir, f"mem-rank{rank}.jsonl")) if m["step"] == s]
        loss, sync = r.get("loss"), r.get("perf/weight_sync_s")
        gap = r.get("sampling/sampling_logp_difference/mean")
        peak = max((m["peak_reserved_gb"] for m in mem), default=float("nan"))
        say(f"step {s}: loss {loss}, weight sync {sync} s, logp diff mean {gap}, reward {r.get('reward')}, "
            f"peak reserved {peak:.1f} GB over {len(mem)} ranks, alloc retries {max((m['num_alloc_retries'] for m in mem), default=0)}")
        if loss is None or not math.isfinite(loss):
            fail(f"step {s} loss is {loss}")
        if sync is None or sync > 60:
            fail(f"step {s} weight sync took {sync} s")
        if gap is None or gap > 0.05:
            fail(f"step {s} trainer-vs-vLLM logp diff mean is {gap} (> 0.05: weight sync or numerics broken)")
        if mem and peak > 175:
            fail(f"step {s} peak reserved {peak:.1f} GB (> 175 GB of 180)")
        checked_steps.add(s)
    if len(checked_steps) >= check_steps and not os.path.exists(os.path.join(run_dir, "WATCHDOG_STEPS_OK")):
        open(os.path.join(run_dir, "WATCHDOG_STEPS_OK"), "w").write("ok\n")
        say(f"first {check_steps} steps OK")

    episodes = read_jsonl(os.path.join(run_dir, "episodes.jsonl"))
    if not episodes_checked and len(episodes) >= 64:
        first = episodes[:64]
        statuses = {s: sum(e["status"] == s for e in first) for s in {e["status"] for e in first}}
        say(f"first 64 episodes: mean reward {sum(e['score'] for e in first) / 64:.3f}, correct "
            f"{sum(e['correct'] for e in first) / 64:.3f}, turns {sum(e['turns'] for e in first) / 64:.2f}, tool calls "
            f"{sum(e['tool_calls'] for e in first) / 64:.2f}, sampled tokens {sum(e['sampled_tokens'] for e in first) / 64:.0f}, "
            f"statuses {statuses}, mean wall {sum(e['end'] - e['start'] for e in first) / 64:.0f} s")
        if sum(e["sampled_tokens"] > 0 for e in first) < 32:
            fail("fewer than 32 of the first 64 episodes sampled any token")
        if all(e["score"] <= -1 for e in first):
            fail("none of the first 64 episodes answered (all scored -1)")
        episodes_checked = True

    last_step = max((r["step"] for r in steps), default=0)
    for ckpt in sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1])):
        n = int(ckpt.rsplit("-", 1)[1])
        if n in merged or last_step <= n:
            continue
        out = os.path.join(merged_root, f"{run_id}-checkpoint-{n}")
        say(f"merging {ckpt} -> {out}")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "merge_lora.py"), ckpt, out, os.environ.get("MODEL", "Qwen/Qwen3.5-35B-A3B")],
            stdout=open(os.path.join(run_dir, f"merge-{n}.log"), "w"), stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""},
        )
        check = json.load(open(os.path.join(out, "merge_check.json"))) if proc.returncode == 0 else {}
        ok = proc.returncode == 0 and check.get("changed", 0) > 0 and check.get("lora_B_nonzero", 0) > 0
        say(f"merge checkpoint-{n}: exit {proc.returncode}, {time.time() - t0:.0f} s, changed {check.get('changed')} of "
            f"{check.get('tensors')} tensors, lora_B nonzero {check.get('lora_B_nonzero')}")
        merged[n] = ok
        if not ok and len(merged) == 1:
            fail(f"first checkpoint merge failed (see merge-{n}.log)")
    time.sleep(30)
