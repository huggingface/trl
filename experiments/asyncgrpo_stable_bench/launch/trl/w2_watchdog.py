"""
In-job checks for `launch/trl/moe_w2.sbatch` (no separate smoke job): watches the run dir while the trainer runs and stops
the trainer with a clear message on the first failure. Writes every verdict to <run_dir>/watchdog.txt.

    python w2_watchdog.py <run_dir> <check_steps> <first_step_timeout_s>

1. First `check_steps` steps: each must log a finite loss, a weight sync under 60 s, a trainer-vs-vLLM logp gap
   (`sampling/sampling_logp_difference/mean`) under 0.05 (35B W1: 0.008-0.015, 9B W2: 0.005-0.007), and peak reserved
   memory under 175 GB on every rank. Step 1 must arrive within `first_step_timeout_s`.
2. Episodes: of the first 32 finished, at least half scored. Episodes past STEP_LIMIT turns are reported, not fatal.
3. Checkpoints: each `checkpoint-N` is merged into a full HF model (`merge_lora.py`, CPU) once the next step has
   logged (so the save is complete). The first merge must succeed with nonzero LoRA weights; later merge failures are
   reported only. Merged models land in stable-bench-runs/trl-moe-ep/merged/<run_id>-checkpoint-N, ready for the eval.
4. Host memory, report only (a local box once grew to 686 GB RSS, ledger all-022; Harbor's singularity env sets no
   per-box limit): every 60 s, MemAvailable and the largest box's RSS (a box = one child process tree of a Harbor server)
   go to <run_dir>/hostmem.csv, plus a watchdog.txt line when MemAvailable < 300 GB or a box is over 50 GB.
"""

import glob
import json
import math
import os
import subprocess
import sys
import threading
import time


run_dir, check_steps, first_timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
run_id = os.path.basename(run_dir)
here = os.path.dirname(os.path.abspath(__file__))
merged_root = "/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/merged"
step_limit = int(os.environ.get("STEP_LIMIT", "30"))
log = open(os.path.join(run_dir, "watchdog.txt"), "a", buffering=1)


def say(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    log.write(line + "\n")
    print(f"[watchdog] {line}", flush=True)


def fail(msg):
    say(f"WATCHDOG FAIL: {msg}. Stopping the trainer.")
    open(os.path.join(run_dir, "WATCHDOG_FAILED"), "w").write(msg + "\n")
    subprocess.run(["pkill", "-TERM", "-u", os.environ["USER"], "-f", "moe_w2.py"])
    time.sleep(60)
    subprocess.run(["pkill", "-KILL", "-u", os.environ["USER"], "-f", "moe_w2.py"])
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


def host_memory():
    """MemAvailable (GB), and the largest box: (RSS GB, pid, command) of the biggest child tree of any Harbor server."""
    avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")) / 1e6
    ppid, rss, cmd = {}, {}, {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            fields = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            ppid[int(pid)] = int(fields[1])
            rss[int(pid)] = int(fields[21]) * os.sysconf("SC_PAGE_SIZE") / 1e9
            cmd[int(pid)] = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
        except (OSError, IndexError, ValueError):
            pass
    children = {}
    for pid, parent in ppid.items():
        children.setdefault(parent, []).append(pid)

    def tree_rss(pid):
        return rss.get(pid, 0) + sum(tree_rss(c) for c in children.get(pid, []))

    servers = [p for p, c in cmd.items() if "harbor serve" in c and "pkill" not in c]
    boxes = [(tree_rss(b), b, cmd.get(b, "")[:100]) for s in servers for b in children.get(s, [])]
    return avail, max(boxes, default=(0.0, 0, "")), len(boxes)


def watch_host_memory():
    path = os.path.join(run_dir, "hostmem.csv")
    if not os.path.exists(path):
        open(path, "w").write("time,mem_available_gb,n_server_children,max_box_rss_gb,max_box_pid,max_box_cmd\n")
    while True:
        try:
            avail, (box_gb, box_pid, box_cmd), n = host_memory()
            open(path, "a").write(f"{time.time():.0f},{avail:.1f},{n},{box_gb:.2f},{box_pid},{box_cmd.replace(',', ' ')}\n")
            if avail < 300 or box_gb > 50:
                say(f"HOST MEMORY (report only): MemAvailable {avail:.0f} GB, largest box {box_gb:.1f} GB (pid {box_pid}: {box_cmd})")
        except Exception as e:  # report-only: never let it stop the watchdog
            say(f"host memory sample failed: {e!r}")
        time.sleep(60)


threading.Thread(target=watch_host_memory, daemon=True).start()
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
    if not episodes_checked and len(episodes) >= 32:
        first = episodes[:32]
        scored = sum(e.get("reward") is not None for e in first)
        over = sum((e.get("n_turns") or 0) > step_limit for e in first)
        say(f"first 32 episodes: {scored} scored, mean reward "
            f"{sum(e['reward'] for e in first if e.get('reward') is not None) / max(scored, 1):.3f}, "
            f"max turns {max((e.get('n_turns') or 0) for e in first)}, {over} over STEP_LIMIT={step_limit}, "
            f"mean wall {sum(e['end'] - e['start'] for e in first) / 32:.0f} s")
        if scored < 16:
            errors = sorted({str(e.get('error'))[:120] for e in first if e.get('reward') is None})[:5]
            fail(f"only {scored} of the first 32 episodes scored; errors: {errors}")
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
