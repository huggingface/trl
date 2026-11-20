"""In-job smoke checks for launch/verl/w2_t2sql.sbatch (no separate smoke job). Polls the run dir every 60 s; on the
first failed check writes the reason to <run_dir>/WATCHDOG, prints it and exits 1 (the launcher then kills training).
Exits 0 once the first steps and the first merged checkpoint are fine. Prints a status line per new step either way.

Checks:
  - stdout.log: CUDA OOM / "Error in _run_prompt" storms before step 1; no step 1 within FIRST_STEP_MIN (default 180) min.
  - episodes: >= 64 finished within 150 min; of the first 256, at most half with reward <= -1 (no answer / unsafe).
  - steps 1-2: peak trainer memory < 172 GB, weight sync < 600 s, |actor/ppo_kl| < 0.05, grad norm finite.
  - first checkpoint: merge_check.json within 60 min of its step's metrics.
Usage: python t2sql_watch.py <run_dir> <save_freq>
"""

import glob
import json
import math
import os
import sys
import time


run_dir, save_freq = sys.argv[1], int(sys.argv[2])
first_step_min = float(os.environ.get("FIRST_STEP_MIN", 180))
t0 = time.time()


def fail(msg):
    msg = f"WATCHDOG {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {msg}"
    open(os.path.join(run_dir, "WATCHDOG"), "w").write(msg + "\n")
    print(msg, flush=True)
    sys.exit(1)


def metrics():
    rows = {}
    path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows[int(r["step"])] = r["data"]
    return rows


def episodes():
    eps = []
    for p in glob.glob(os.path.join(run_dir, "episodes", "*.jsonl")):
        for line in open(p):
            try:
                eps.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return sorted(eps, key=lambda e: e["end"])


episodes_ok = steps_ok = False
ckpt_ok = save_freq <= 0
ckpt_seen_at = None
printed = 0
merged = f"/fsx/amine_dirhoussi/stable-bench-runs/verl/merged/{os.path.basename(run_dir)}-checkpoint-{save_freq}"
while True:
    time.sleep(60)
    m = metrics()
    for step in sorted(s for s in m if s > printed):
        d = m[step]
        print(f"step {step} t+{(time.time() - t0) / 60:.0f}min: score {d.get('critic/score/mean')}, step {d.get('timing_s/step')} s, "
              f"gen-wait {d.get('timing_s/gen')} s, update {d.get('timing_s/update_actor')} s, sync {d.get('timing_s/update_weights')} s, "
              f"mem {d.get('actor/perf/max_memory_allocated_gb')} GB, ppo_kl {d.get('actor/ppo_kl')}, kl_loss {d.get('actor/kl_loss')}, "
              f"grad {d.get('actor/grad_norm')}, staleness max {d.get('training/off_policy/trajectory_staleness/max')}", flush=True)
        printed = step
    if steps_ok and episodes_ok and ckpt_ok:
        continue
    eps = episodes()
    if not episodes_ok:
        first = eps[:256]
        if len(first) >= 128 and sum(e["score"] <= -1 for e in first) > len(first) / 2:
            fail(f"{sum(e['score'] <= -1 for e in first)} of the first {len(first)} episodes scored <= -1 (no answer / unsafe)")
        if len(eps) >= 64:
            episodes_ok = True
            print(f"WATCHDOG ok: {len(eps)} episodes, first 64: score {sum(e['score'] for e in eps[:64]) / 64:.3f}, "
                  f"correct {sum(e['correct'] for e in eps[:64]) / 64:.3f}, turns {sum(e['turns'] for e in eps[:64]) / 64:.1f}, "
                  f"sampled tok {sum(e['sampled_tokens'] for e in eps[:64]) / 64:.0f}", flush=True)
        elif time.time() - t0 > 150 * 60:
            fail(f"only {len(eps)} episodes after 150 min")
    if not m:
        log = os.path.join(run_dir, "stdout.log")
        text = open(log, errors="replace").read()[-300000:] if os.path.exists(log) else ""
        if "CUDA out of memory" in text or "OutOfMemoryError" in text:
            fail("CUDA OOM in stdout.log before step 1")
        if text.count("Error in _run_prompt") >= 20:
            fail(f"{text.count('Error in _run_prompt')} 'Error in _run_prompt' in stdout.log before step 1")
        if time.time() - t0 > first_step_min * 60:
            fail(f"no step 1 after {first_step_min:.0f} min")
    if not steps_ok and 1 in m:
        for step in (1, 2):
            d = m.get(step)
            if d is None:
                break
            mem, sync = d.get("actor/perf/max_memory_allocated_gb"), d.get("timing_s/update_weights")
            kl, gn = d.get("actor/ppo_kl"), d.get("actor/grad_norm")
            if mem is not None and mem > 172:
                fail(f"step {step}: peak trainer memory {mem:.1f} GB")
            if sync is not None and sync > 600:
                fail(f"step {step}: weight sync {sync:.0f} s")
            if kl is not None and abs(kl) > 0.05:
                fail(f"step {step}: actor/ppo_kl {kl:.4f} (trainer vs vLLM logprobs disagree)")
            if gn is not None and not math.isfinite(gn):
                fail(f"step {step}: grad norm {gn}")
        else:
            steps_ok = True
            print("WATCHDOG ok: steps 1-2", flush=True)
    if not ckpt_ok and save_freq in m:
        ckpt_seen_at = ckpt_seen_at or time.time()
        if os.path.exists(os.path.join(merged, "merge_check.json")):
            ckpt_ok = True
            print(f"WATCHDOG ok: merged {merged}", flush=True)
        elif time.time() - ckpt_seen_at > 60 * 60:
            fail(f"no merged checkpoint at {merged} 60 min after step {save_freq}")
    if steps_ok and episodes_ok and ckpt_ok:
        print("WATCHDOG: all smoke checks passed (still printing steps)", flush=True)
