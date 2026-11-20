"""verl native logs -> results/raw/<run_id>/ (collect/SCHEMA.md).

Inputs, all in the run dir written by launch/verl/w1.sbatch:
  metrics*.jsonl  verl's file logger, one {"step", "data": {metric: value}} row per global step (= optimizer step here)
  stdout.log      driver stdout, every line prefixed with a unix timestamp (gives the wall clock of each step)
  config.yaml     the Hydra config the run used; gpu.csv; verl_sha; nodes
Usage: python collect/verl.py <run_dir> [--job-log <slurm .out>] [--status completed|failed|killed]
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil

import yaml


BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_metrics(run_dir):
    rows = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "metrics*.jsonl")), key=os.path.getmtime):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows[int(r["step"])] = r["data"]  # a resumed attempt overwrites the steps it redid
    return rows


def step_times(run_dir):
    """Wall clock of the console line `step:N - ...` that verl prints when step N is logged."""
    times = {}
    pat = re.compile(r"^(\d+\.\d+) .*?\bstep:(\d+) - ")
    with open(os.path.join(run_dir, "stdout.log"), errors="replace") as f:
        for line in f:
            m = pat.match(line)
            if m:
                times[int(m.group(2))] = float(m.group(1))
    return times


def job_epochs(job_log):
    start = end = None
    if job_log and os.path.exists(job_log):
        with open(job_log, errors="replace") as f:
            for line in f:
                if line.startswith("JOB_START_EPOCH") and start is None:
                    start = float(line.split()[1])
                elif line.startswith("JOB_END_EPOCH"):
                    end = float(line.split()[1])
    return start, end


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--job-log")
    ap.add_argument("--status", default="completed")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    run_id = os.path.basename(run_dir)
    out = os.path.join(BENCH, "results", "raw", run_id)
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg_text = f.read()
    cfg = yaml.safe_load(cfg_text)
    base = os.path.join(run_dir, "config_base.yaml")
    if any(d != "_self_" and d != "ppo_trainer" for d in cfg.get("defaults", [])) and os.path.exists(base):
        cfg = merge(yaml.safe_load(open(base)), cfg)  # an arm config that extends w1-<size>.yaml
    config_name = (
        open(os.path.join(run_dir, "config_name")).read().strip()
        if os.path.exists(os.path.join(run_dir, "config_name"))
        else None
    )
    aro = cfg["actor_rollout_ref"]
    n = aro["rollout"]["n"]
    # One verl global step = parameter_sync_step optimizer steps of ppo_mini_batch_size prompts each. verl logs one
    # metrics row per global step (mini-batch metrics are averaged), so a row covers psync optimizer steps and its
    # `step` is the optimizer-step index at the end of that global step.
    psync = cfg["trainer"]["v1"]["separate_async"]["parameter_sync_step"]
    rollouts_per_step = cfg["data"]["train_batch_size"] * n

    metrics = load_metrics(run_dir)
    times = step_times(run_dir)
    with open(os.path.join(out, "steps.jsonl"), "w") as f:
        for step in sorted(metrics):
            d = metrics[step]
            len_mean = get(d, "response_length/mean")
            row = {
                "step": step * psync,
                "global_step": step,
                "time": times.get(step),
                "step_time": get(d, "timing_s/step"),
                "train_compute_time": get(d, "timing_s/update_actor"),
                # V1 separate_async: time the trainer waits for the next trainable batch (docs/advance/v1_async_trainer.md)
                "wait_for_rollouts_time": get(d, "timing_s/gen"),
                "weight_sync_time": get(d, "timing_s/update_weights"),
                "num_rollouts": rollouts_per_step,
                "gen_tokens": None if len_mean is None else len_mean * rollouts_per_step,
                # perf/total_num_tokens = prompt + response tokens of the step's batch (verl packs, no pad tokens)
                "trained_tokens": get(d, "perf/total_num_tokens"),
                "padding_fraction": None,
                "reward_mean": get(d, "critic/score/mean"),
                "reward_std": None,
                "frac_zero_adv_groups": None,
                "completion_len_mean": len_mean,
                "completion_len_max": get(d, "response_length/max"),
                "truncated_fraction": get(d, "response_length/clip_ratio"),
                "staleness_mean": get(d, "training/off_policy/trajectory_staleness/mean"),
                "staleness_max": get(
                    d,
                    "training/off_policy/trajectory_staleness_worst/max",
                    "training/off_policy/trajectory_staleness/max",
                ),
                # bypass mode: PPO's old logprobs are vLLM's, so actor/ppo_kl = mean(logp_vllm - logp_trainer) over the
                # step's tokens (micro-batch mean). It mixes engine mismatch with policy lag (staleness).
                "logprob_mismatch_kl": get(d, "actor/ppo_kl"),
                "grad_norm": get(d, "actor/grad_norm"),
                "lr": get(d, "actor/lr"),
                "peak_mem_gb": get(d, "actor/perf/max_memory_allocated_gb", "perf/max_memory_allocated_gb"),
            }
            f.write(json.dumps(row) + "\n")

    if os.path.exists(os.path.join(run_dir, "gpu.csv")):
        shutil.copy(os.path.join(run_dir, "gpu.csv"), os.path.join(out, "gpu.csv"))

    start, end = job_epochs(args.job_log)
    nodes = (
        open(os.path.join(run_dir, "nodes")).read().split() if os.path.exists(os.path.join(run_dir, "nodes")) else []
    )
    sha = (
        open(os.path.join(run_dir, "verl_sha")).read().strip()
        if os.path.exists(os.path.join(run_dir, "verl_sha"))
        else None
    )
    m = re.search(r"-(\d+)$", run_id)
    parts = run_id.split("-")
    trainer = cfg["trainer"]
    meta = {
        "run_id": run_id,
        "lib": "verl",
        "lib_version": "0.9.1",
        "lib_sha": sha,
        "trl_sha": None,
        "workload": parts[1],
        "size": parts[2],
        "model": aro["model"]["path"],
        "config_file": f"configs/verl/{config_name or parts[1] + '-' + parts[2]}.yaml",
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "slurm_job_id": int(m.group(1)) if m else None,
        "nodes": nodes,
        "gpus_total": 8 * len(nodes) if nodes else 8,
        "gpus_inference": aro["rollout"]["n_gpus_per_node"] * aro["rollout"]["nnodes"],
        "gpus_train": trainer["n_gpus_per_node"] * trainer["nnodes"],
        "inference_engine": "vllm 0.24.0",
        "train_backend": aro["actor"]["strategy"],
        "master_dtype": "fp32",
        "max_staleness": trainer["v1"]["sampler"]["max_off_policy_threshold"],
        "start_time": start,
        "end_time": end,
        "gpu_hours": None if start is None or end is None else 8 * (end - start) / 3600,
        "status": args.status,
        "notes": "verl V1 separate_async, hybrid_engine=false, parameter_sync_step=%d (one steps.jsonl row per verl global step = %d optimizer "
        "steps); max_staleness is in model versions (drop strategy)" % (psync, psync),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out}: {len(metrics)} steps")


if __name__ == "__main__":
    main()
