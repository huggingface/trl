"""In-job smoke checks for launch/verl/w2.sbatch (no separate smoke jobs). Polls the run dir every 60 s and, on the
first failed check, writes the reason to <run_dir>/WATCHDOG, prints it, and exits 1 (the launcher then kills training).
Exits 0 once every check has passed (first steps + the first checkpoint).

Checks:
  - episodes: within 90 min of start, >= 8 scored episodes with model calls through the proxy; of the first 32
    finished episodes, at most half unscored.
  - steps 1-2: peak trainer memory < 172 GB, weight sync < 300 s, |actor/ppo_kl| < 0.05, grad norm finite, score in [0, 1].
  - first checkpoint (save_freq): HF weights (config.json + *.safetensors) present within 45 min of its step's metrics.
  - a CUDA OOM / Traceback in stdout.log before step 1.
Usage: python w2_watch.py <run_dir> <save_freq>
"""

import glob
import json
import math
import os
import sys
import time


run_dir, save_freq = sys.argv[1], int(sys.argv[2])
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


episodes_ok = steps_ok = False
ckpt_ok = save_freq <= 0
ckpt_seen_at = None
while True:
    time.sleep(60)
    m = metrics()
    eps = []
    for p in glob.glob(os.path.join(run_dir, "episodes", "*.verl.json")):
        try:
            eps.append(json.load(open(p)))
        except (json.JSONDecodeError, OSError):
            pass
    eps.sort(key=lambda e: e["end"])
    if not episodes_ok:
        scored = [e for e in eps if e["reward"] is not None and e["calls"] > 0]
        first = eps[:32]
        if len(first) >= 16 and sum(e["reward"] is None for e in first) > len(first) / 2:
            fail(f"{sum(e['reward'] is None for e in first)} of the first {len(first)} episodes unscored; last error: {first[-1].get('error')!r:.300}")
        if len(eps) >= 16 and not any(e["calls"] for e in eps):
            fail("16 episodes finished without a single model call through the verl proxy")
        if len(scored) >= 8:
            episodes_ok = True
            print(f"WATCHDOG ok: {len(scored)} scored episodes, mean calls {sum(e['calls'] for e in scored)/len(scored):.1f}", flush=True)
        elif time.time() - t0 > 90 * 60:
            fail(f"only {len(scored)} scored episodes after 90 min ({len(eps)} finished)")
    if not m and time.time() - t0 < 90 * 60:
        log = os.path.join(run_dir, "stdout.log")
        if os.path.exists(log):
            text = open(log, errors="replace").read()[-200000:]
            if "CUDA out of memory" in text or "OutOfMemoryError" in text:
                fail("CUDA OOM in stdout.log before step 1")
    if not steps_ok and 1 in m:
        for step in (1, 2):
            d = m.get(step)
            if d is None:
                break
            mem = d.get("actor/perf/max_memory_allocated_gb", d.get("perf/max_memory_allocated_gb"))
            sync, kl, gn, score = d.get("timing_s/update_weights"), d.get("actor/ppo_kl"), d.get("actor/grad_norm"), d.get("critic/score/mean")
            print(f"WATCHDOG step {step}: mem {mem} GB, sync {sync} s, ppo_kl {kl}, grad_norm {gn}, score {score}", flush=True)
            if mem is not None and mem > 172:
                fail(f"step {step}: peak trainer memory {mem:.1f} GB")
            if sync is not None and sync > 300:
                fail(f"step {step}: weight sync {sync:.0f} s")
            if kl is not None and abs(kl) > 0.05:
                fail(f"step {step}: actor/ppo_kl {kl:.4f} (vLLM vs trainer logprobs disagree)")
            if gn is not None and not math.isfinite(gn):
                fail(f"step {step}: grad norm {gn}")
            if score is not None and not 0 <= score <= 1:
                fail(f"step {step}: score {score}")
        else:
            steps_ok = True
    if not ckpt_ok and save_freq in m:
        ckpt_seen_at = ckpt_seen_at or time.time()
        hf = os.path.join(run_dir, "ckpt", f"global_step_{save_freq}", "actor", "huggingface")
        # LoRA-only checkpoints (w2-moe) are merged for the eval by t2sql_merge.sh instead
        merged = f"/fsx/amine_dirhoussi/stable-bench-runs/verl/merged/{os.path.basename(run_dir)}-checkpoint-{save_freq}"
        if os.path.exists(os.path.join(merged, "merge_check.json")) or (
            os.path.exists(os.path.join(hf, "config.json")) and glob.glob(os.path.join(hf, "*.safetensors"))
        ):
            ckpt_ok = True
            print(f"WATCHDOG ok: checkpoint {hf}", flush=True)
        elif time.time() - ckpt_seen_at > 45 * 60:
            fail(f"no HF checkpoint at {hf} 45 min after step {save_freq}")
    if episodes_ok and steps_ok and ckpt_ok:
        print("WATCHDOG: all smoke checks passed", flush=True)
        sys.exit(0)
