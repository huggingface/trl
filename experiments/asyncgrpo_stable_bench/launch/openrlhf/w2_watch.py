"""In-job fail-fast checks for launch/openrlhf/w2.sbatch. Polls every 60 s; on the first failed check writes the reason
to <run_dir>/WATCHDOG, prints it and exits 1 (the launcher then kills the trainer).
  - no step 1 within 90 min of start; CUDA OOM or Traceback-with-ActorDied in the job log
  - non-finite policy_loss / actor_grad_norm
  - collapse: 5 consecutive steps with rollout/reward_mean < 0.05, or a step with more than half its episodes unscored
Usage: python w2_watch.py <run_dir> <job_log>
"""
import importlib.util, math, os, sys, time
# collect/openrlhf.py by path (its module name would shadow the openrlhf package)
spec = importlib.util.spec_from_file_location("collect_openrlhf", os.path.join(os.path.dirname(__file__), "../../collect/openrlhf.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
read_tb = mod.read_tb

run_dir, job_log = sys.argv[1:3]
t0 = time.time()


def fail(msg):
    msg = f"WATCHDOG {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {msg}"
    open(os.path.join(run_dir, "WATCHDOG"), "w").write(msg + "\n")
    print(msg, flush=True)
    sys.exit(1)


seen = 0
while True:
    time.sleep(60)
    text = open(job_log, errors="replace").read()[-500000:] if os.path.exists(job_log) else ""
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        fail("CUDA OOM in the job log")
    if "ActorDiedError" in text or "RayActorError" in text:
        fail("a Ray actor died")
    try:
        values, _ = read_tb(os.path.join(run_dir, "tb"))
    except Exception:
        continue
    steps = sorted(values)
    if not steps:
        if time.time() - t0 > 90 * 60:
            fail("no step 1 after 90 min")
        continue
    for s in steps[seen:]:
        v = values[s]
        print(f"WATCHDOG step {s}: reward {v.get('rollout/reward_mean')} loss {v.get('policy_loss')} grad {v.get('actor_grad_norm')} "
              f"vllm_kl {v.get('vllm_kl')} unscored {v.get('w2_unscored')} calls {v.get('w2_calls')}", flush=True)
        for k in ("policy_loss", "actor_grad_norm"):
            if k in v and not math.isfinite(v[k]):
                fail(f"step {s}: {k} = {v[k]}")
        if v.get("w2_unscored", 0) > 0.5:
            fail(f"step {s}: {v['w2_unscored']:.2f} of episodes unscored")
    seen = len(steps)
    rew = [values[s].get("rollout/reward_mean") for s in steps[-5:]]
    if len(steps) >= 5 and all(r is not None and r < 0.05 for r in rew):
        fail(f"collapse: reward_mean < 0.05 for steps {steps[-5:]}")
