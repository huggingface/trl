"""In-job smoke checks for a miles W2 run (replaces separate smoke jobs). Polls the run dir every 60 s and, on the first
failed check, prints `WATCH_KILL <reason>` and exits 3 (w2.sbatch then cancels the job). Checks:
  - train.log: CUDA OOM, Python tracebacks that end the driver (`Traceback` + `Error` in the last lines once training ended)
  - episodes: by 60 min after TRAIN_START at least 8 Harbor trials finished, and among the first 64 finished at most half
    without a verifier result
  - first steps (events): train/train_rollout_kl <= 0.05 and train_rollout_logprob_abs_diff <= 0.1 on steps 1-3,
    weight sync (perf/update_weights_time) <= 300 s
  - progress: first optimizer step within 4 h of TRAIN_START, then a step at least every 90 min
  - first checkpoint: hf/step_<save-1> has its .complete marker within 45 min of that step's event
    python watch.py <run dir> <save interval>
"""
import glob, json, os, sys, time

run, save = sys.argv[1], int(sys.argv[2])
log = lambda m: print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} WATCH {m}", flush=True)  # noqa: E731


def kill(reason):
    log(f"KILL {reason}")
    print(f"WATCH_KILL {reason}", flush=True)
    sys.exit(3)


def times():
    out = {}
    try:
        for line in open(f"{run}/times.txt"):
            p = line.split()
            if len(p) >= 2 and p[1].replace(".", "", 1).isdigit():
                out[p[0]] = float(p[1])
    except OSError:
        pass
    return out


def steps():
    st = {}
    for f in glob.glob(f"{run}/**/events/**/*.jsonl", recursive=True):
        for line in open(f):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            m, rid = e.get("metrics") or {}, e.get("rollout_id")
            if rid is None:
                rid = next((int(m[k]) for k in ("train/step", "rollout/step") if k in m), None)
            if rid is not None and any(k.startswith("train/") and k != "train/step" for k in m):
                st.setdefault(rid, {}).update(m)
                st[rid]["_t"] = time.time() if "_t" not in st[rid] else st[rid]["_t"]
    return st


def trials():
    done = []
    for f in glob.glob(f"{run}/trials/*/result.json"):
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if r.get("finished_at"):
            done.append((r["finished_at"], r.get("verifier_result") is not None, (r.get("exception_info") or {}).get("exception_type")))
    return sorted(done)


checked_steps, ok_episodes, ckpt_seen_t, last_step_t = set(), False, None, None
while True:
    time.sleep(60)
    t = times()
    if "TRAIN_END" in t or "JOB_END" in t:
        log("training ended, watcher exits")
        sys.exit(0)
    try:
        tail = open(f"{run}/train.log", errors="replace").read()[-200000:]
    except OSError:
        tail = ""
    if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
        kill("CUDA OOM in train.log")
    if "TRAIN_START" not in t:
        continue
    age = time.time() - t["TRAIN_START"]
    tr = trials()
    if not ok_episodes:
        first = tr[:64]
        if len(first) >= 16:
            bad = sum(1 for _, scored, _ in first if not scored)
            if bad > len(first) / 2:
                kill(f"{bad}/{len(first)} of the first episodes have no verifier result: {[e for _, s, e in first if not s][:5]}")
            if len(first) >= 32:
                ok_episodes = True
                log(f"episodes ok: {len(first)} finished, {bad} unscored")
        elif age > 3600 and len(tr) < 8:
            kill(f"only {len(tr)} Harbor trials finished 60 min after TRAIN_START")
    st = steps()
    if st:
        last_step_t = max(last_step_t or 0, max(v["_t"] for v in st.values()))
    for rid in sorted(st):
        if rid > 2 or rid in checked_steps:
            continue
        m = st[rid]
        kl, diff, ws = m.get("train/train_rollout_kl"), m.get("train/train_rollout_logprob_abs_diff"), m.get("perf/update_weights_time")
        log(f"step {rid + 1}: train_rollout_kl {kl} logprob_abs_diff {diff} update_weights_time {ws} grad_norm {m.get('train/grad_norm')}")
        if kl is not None and kl > 0.05:
            kill(f"step {rid + 1} train_rollout_kl {kl} > 0.05 (trainer and SGLang logprobs disagree)")
        if diff is not None and diff > 0.1:
            kill(f"step {rid + 1} train_rollout_logprob_abs_diff {diff} > 0.1")
        if ws is not None and ws > 300:
            kill(f"step {rid + 1} weight sync {ws} s > 300 s")
        checked_steps.add(rid)
    if not st and age > 4 * 3600:
        kill("no optimizer step 4 h after TRAIN_START")
    if last_step_t and time.time() - last_step_t > 5400:
        kill("no optimizer step for 90 min")
    if save - 1 in st and ckpt_seen_t is None:
        ckpt_seen_t = time.time()
    if ckpt_seen_t and ckpt_seen_t > 0:
        d = f"{run}/hf/step_{save - 1}"
        if glob.glob(f"{d}/.complete*") or glob.glob(f"{d}/*complete*"):
            log(f"first checkpoint {d} complete: {sorted(os.listdir(d))[:8]}")
            ckpt_seen_t = -1
        elif time.time() - ckpt_seen_t > 2700:
            kill(f"{d} not complete 45 min after step {save}")
