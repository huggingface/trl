"""OpenRLHF native logs -> results/raw/<run_id>/ (collect/SCHEMA.md).

    python collect/openrlhf.py /fsx/amine_dirhoussi/stable-bench-runs/openrlhf/<run_id> [--status completed]

Reads the TensorBoard events the trainer writes (`--logger.tensorboard_dir <run dir>/tb`, one scalar per status key per
step, with the wall time of the write), the copied flag file `config.sh`, `job_times.txt` and `gpu.csv` from the
launch script. Raw values only: a field OpenRLHF doesn't log is null.
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "/fsx/amine_dirhoussi/stable-bench-src/openrlhf"


def read_tb(tb_dir):
    """{step: {tag: value}} and {step: wall time of the last scalar written for that step}."""
    values, times = defaultdict(dict), defaultdict(float)
    for path in sorted(glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents.*"), recursive=True)):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if not tag.startswith("train/"):
                continue
            for ev in acc.Scalars(tag):
                values[ev.step][tag[len("train/") :]] = ev.value
                times[ev.step] = max(times[ev.step], ev.wall_time)
    return values, times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--status", default=None, help="completed / failed / killed (default: from job_times.txt rc)")
    args = p.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    run_id = os.path.basename(run_dir)
    _, workload, size, *_, job_id = run_id.split("-")
    out = os.path.join(BENCH, "results", "raw", run_id)
    os.makedirs(out, exist_ok=True)

    values, times = read_tb(os.path.join(run_dir, "tb"))
    run_args = open(os.path.join(run_dir, "args.txt")).read().split()
    # Without partial rollout the staleness is structural (see below). With it, a sample can mix policy versions and
    # OpenRLHF doesn't log which, so it is null.
    partial = "--train.partial_rollout_enable" in run_args
    rows = []
    for step in sorted(values):
        v = values[step]
        n = v.get("rollout/num_samples")
        gen_len = v.get("rollout/response_length_mean")
        tot_len = v.get("total_length")  # sample-weighted mean of prompt + response tokens over trained samples
        rows.append(
            {
                "step": step,
                "time": times[step],
                "step_time": v.get(
                    "timing/step_total"
                ),  # wall time since the previous step ended, queue wait included
                "train_compute_time": v.get("timing/ppo_train"),
                "wait_for_rollouts_time": None,
                # timing/broadcast includes waiting for the VLLMLock, i.e. for the in-flight next batch to finish
                # generating, so it is not the sync alone: kept raw as broadcast_incl_lock_wait below
                "weight_sync_time": None,
                "num_rollouts": n,
                "gen_tokens": gen_len * n if gen_len is not None and n is not None else None,
                "trained_tokens": tot_len * n if tot_len is not None and n is not None else None,
                "padding_fraction": None,
                "reward_mean": v.get("rollout/reward_mean"),
                "reward_std": v.get("rollout/reward_std"),
                "frac_zero_adv_groups": None,
                "completion_len_mean": gen_len,
                "completion_len_max": None,
                "truncated_fraction": v.get("rollout/truncated_rate"),
                # Structural, not logged: the VLLMLock + queue size 1 make batch 1 on-policy and every later batch
                # exactly one policy version old (ppo_trainer_async.py).
                "staleness_mean": None if partial else 0 if step == 1 else 1,
                "staleness_max": None if partial else 0 if step == 1 else 1,
                # mean over action tokens of (log pi_vllm - log pi_old_trainer), logged when IS correction is on
                "logprob_mismatch_kl": v.get("vllm_kl"),
                "grad_norm": v.get("actor_grad_norm"),
                "lr": v.get("actor_lr"),
                "peak_mem_gb": None,
                # extra native fields, kept raw
                "generation_time": v.get("timing/generation"),
                "broadcast_incl_lock_wait": v.get("timing/broadcast"),
                "make_experience_time": v.get("timing/make_experience"),
                "is_filter_ratio": v.get("is_filter_ratio"),
                "ppo_clip_ratio": v.get("ppo_clip_ratio"),
                "policy_loss": v.get("policy_loss"),
            }
        )
    with open(os.path.join(out, "steps.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if os.path.exists(os.path.join(run_dir, "gpu.csv")):
        shutil.copy(os.path.join(run_dir, "gpu.csv"), os.path.join(out, "gpu.csv"))

    jt = {}
    if os.path.exists(os.path.join(run_dir, "job_times.txt")):
        for line in open(os.path.join(run_dir, "job_times.txt")):
            parts = line.split()
            jt[parts[0]] = float(parts[1])
            if parts[0] == "JOB_END":
                jt["rc"] = int(parts[2].split("=")[1])
    start, end = jt.get("JOB_START"), jt.get("JOB_END")
    status = args.status or ("completed" if jt.get("rc") == 0 else "failed" if "rc" in jt else "killed")
    config = open(os.path.join(run_dir, "config_path.txt")).read().strip()
    used_config = os.path.join(run_dir, "args.txt")  # the resolved flag list the run was launched with
    sacct = subprocess.run(["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "NodeList"], capture_output=True, text=True)
    nodes = sacct.stdout.strip().splitlines()[:1]
    freeze = open(os.path.join(BENCH, "envs", "openrlhf", "freeze.txt")).read()
    vllm_version = next((ln.split("==")[1] for ln in freeze.splitlines() if ln.startswith("vllm==")), None)
    meta = {
        "run_id": run_id,
        "lib": "openrlhf",
        "lib_version": open(os.path.join(SRC, "version.txt")).read().strip(),
        "lib_sha": subprocess.run(
            ["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "trl_sha": None,
        "workload": workload,
        "size": size,
        "model": {"S": "Qwen/Qwen3.5-2B", "M": "Qwen/Qwen3.5-9B"}[size],
        "config_file": config,
        "config_sha256": hashlib.sha256(open(used_config, "rb").read()).hexdigest(),
        "slurm_job_id": job_id,
        "nodes": nodes,
        "gpus_total": 8,
        "gpus_inference": 4,
        "gpus_train": 4,
        "inference_engine": f"vllm {vllm_version}",
        "train_backend": "deepspeed-zero2",
        "master_dtype": "fp32",
        "max_staleness": None if partial else 1,
        "start_time": start,
        "end_time": end or (rows[-1]["time"] if rows else None),
        "gpu_hours": 8 * ((end or (rows[-1]["time"] if rows else start)) - start) / 3600 if start else None,
        "status": status,
        "notes": "step_time = OpenRLHF timing/step_total (wall time between consecutive trainer steps). "
        "staleness is structural (async queue 1 + VLLMLock), not logged. peak_mem_gb not logged: see gpu.csv.",
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{len(rows)} steps -> {out}")


if __name__ == "__main__":
    main()
