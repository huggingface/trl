"""SkyRL native logs -> results/raw/<run_id>/ (collect/SCHEMA.md).

Reads, from the run dir `/fsx/amine_dirhoussi/stable-bench-runs/skyrl/<run_id>/`:
    tb/            tensorboard events written by `trainer.logger=tensorboard` (one scalar per metric per step;
                   the event wall time is when the step's metrics were logged, i.e. step end)
    gpu.csv        1 Hz nvidia-smi sampler from launch/skyrl/common.sh
    run_info.txt   config path + sha256, exit status
    job_times.txt  job start/end (unix)
    train.log      only for the vLLM version line

    trials.tgz     W2 only: the Harbor trials dir (launch/skyrl/w2.sbatch archives it at job end), read by collect/agentic.py

Writes meta.json, steps.jsonl and gpu.csv; for W2 also episodes.jsonl and the W2 step fields (collect/SCHEMA.md W2), with
trained / loss / forwarded tokens from the `w2/*` scalars that launch/skyrl/skyrl_w2_main.py logs. No spans.jsonl /
mem.jsonl / rollouts.jsonl: SkyRL does not log them.
Usage: python collect/skyrl.py <run_id> [--status completed|failed|killed]
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict

import agentic
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = "/fsx/amine_dirhoussi/stable-bench-runs/skyrl"
BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "/fsx/amine_dirhoussi/stable-bench-src/skyrl"


def load_scalars(tb_dir):
    acc = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    acc.Reload()
    by_step = defaultdict(dict)
    wall = defaultdict(float)
    for tag in acc.Tags()["scalars"]:
        for ev in acc.Scalars(tag):
            by_step[ev.step][tag] = ev.value
            wall[ev.step] = max(wall[ev.step], ev.wall_time)
    return by_step, wall


def val(m, *keys):
    for k in keys:
        if k in m and m[k] is not None and not (isinstance(m[k], float) and math.isnan(m[k])):
            return float(m[k])
    return None


def mul(a, b):
    return None if a is None or b is None else a * b


def sub(a, *bs):
    if a is None or any(b is None for b in bs):
        return None
    return a - sum(bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--status", default=None)
    args = ap.parse_args()
    run_dir = os.path.join(RUNS, args.run_id)
    out = os.path.join(BENCH, "results", "raw", args.run_id)
    os.makedirs(out, exist_ok=True)

    by_step, wall = load_scalars(os.path.join(run_dir, "tb"))
    n_gpus_train = 4
    workload = args.run_id.split("-")[1]
    episodes = []
    if workload == "w2":
        # trials.tgz, plus trials-r<N>.tgz per requeued attempt
        recs = [
            r
            for t in sorted(os.listdir(run_dir))
            if t.startswith("trials") and t.endswith(".tgz")
            for r in agentic.episodes_from_trials(os.path.join(run_dir, t))
        ]
        for r in recs:
            # terminus-2 stops cleanly at max_turns (30 calls): a budget hit, like the agent timeout
            r["budget_stops"] = r["n_turns"] >= 30
        episodes = sorted((agentic.normalize(r) for r in recs), key=lambda e: e["end"])
        with open(os.path.join(out, "episodes.jsonl"), "w") as f:
            for e in episodes:
                f.write(json.dumps(e) + "\n")

    rows = []
    times0 = dict(line.split("=") for line in open(os.path.join(run_dir, "job_times.txt")).read().split())
    prev_wall = float(times0["JOB_START"])
    for step in sorted(by_step):
        m = by_step[step]
        step_time = val(m, "timing/step")
        train = val(m, "timing/run_training", "timing/train_critic_and_policy")
        convert = val(m, "timing/convert_to_training_input")
        sync = val(m, "timing/sync_weights")
        # Not logged directly: the part of the step not spent converting, training or syncing is waiting for the
        # mini-batch to be generated.
        wait = sub(step_time, convert or 0.0, train, sync)
        if workload == "w2":
            row_w2 = {
                "trained_tokens": val(m, "w2/trained_tokens"),
                "loss_tokens": val(m, "w2/loss_tokens"),
                "forwarded_tokens": val(m, "w2/forwarded_tokens"),
                "prefix_share_ratio": val(m, "w2/prefix_share_ratio"),
                "num_rollouts": val(m, "w2/samples_before_merge"),
                **agentic.step_fields(episodes, prev_wall, wall[step]),
            }
            prev_wall = wall[step]
        rows.append(
            {
                "step": step,
                "time": wall[step],
                "step_time": step_time,
                "train_compute_time": train,
                "wait_for_rollouts_time": wait,
                "weight_sync_time": sync,
                "num_rollouts": val(m, "generate/batch_num_seq"),
                "gen_tokens": mul(val(m, "generate/avg_assistant_tokens"), val(m, "generate/batch_num_seq")),
                # Completion tokens only: generate/avg_num_tokens is len(response) (generators/utils.py), and SkyRL logs no
                # prompt length for W1. Every sequence of the step goes through the backward, zero-advantage ones included.
                "trained_tokens": mul(val(m, "generate/avg_num_tokens"), val(m, "generate/batch_num_seq")),
                # With remove_microbatch_padding=false (W1) every micro-batch keeps the step's global padded width
                # (max prompt + max response; TensorBatch.chunk doesn't trim), so this is what the trainer forwards.
                "forwarded_tokens_padded": mul(
                    val(m, "generate/batch_padded_seq_len"), val(m, "generate/batch_num_seq")
                ),
                "padding_fraction": None,
                "reward_mean": val(m, "reward/avg_raw_reward"),
                "reward_std": None,
                "frac_zero_adv_groups": None,
                "completion_len_mean": val(m, "generate/avg_assistant_tokens"),
                "completion_len_max": val(m, "generate/max_assistant_tokens"),
                "truncated_fraction": None,
                "staleness_mean": val(m, "async/staleness_mean"),
                "staleness_max": val(m, "async/staleness_max"),
                # SkyRL logs mean |logp_train - logp_vllm| (policy/minibatch_rollout_logprobs_abs_diff_mean), not a KL; kept in "native".
                "logprob_mismatch_kl": None,
                "grad_norm": val(m, "policy/grad_norm"),
                "lr": val(m, "policy/policy_lr", "policy/lr"),
                "peak_mem_gb": None,
                "native": m,
            }
        )
        if workload == "w2":
            rows[-1].update(row_w2)
    with open(os.path.join(out, "steps.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    gpu_src = os.path.join(run_dir, "gpu.csv")
    if os.path.exists(gpu_src):
        shutil.copy(gpu_src, os.path.join(out, "gpu.csv"))

    info = open(os.path.join(run_dir, "run_info.txt")).read()
    config_file = re.search(r"CONFIG=(\S+)", info).group(1)
    exit_code = re.search(r"EXIT=(\d+)", info)
    times = dict(line.split("=") for line in open(os.path.join(run_dir, "job_times.txt")).read().split())
    start = float(times["JOB_START"])
    end = float(times["JOB_END"]) if "JOB_END" in times else (max(wall.values()) if wall else start)
    log = (
        open(os.path.join(run_dir, "train.log"), errors="replace").read()
        if os.path.exists(os.path.join(run_dir, "train.log"))
        else ""
    )
    vllm = re.search(r"vllm[ =]+(?:version )?v?(\d+\.\d+\.\d+)", log, re.I)
    sha = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    job_id = args.run_id.rsplit("-", 1)[-1]
    nodes = subprocess.run(
        ["sacct", "-nXj", job_id, "-o", "NodeList%200"], capture_output=True, text=True
    ).stdout.split()
    status = args.status or (
        "completed" if exit_code and exit_code.group(1) == "0" else ("failed" if exit_code else "killed")
    )
    meta = {
        "run_id": args.run_id,
        "lib": "skyrl",
        "lib_version": "main",
        "lib_sha": sha,
        "trl_sha": None,
        "workload": workload,
        "size": args.run_id.split("-")[2],
        "model": re.search(r"trainer\.policy\.model\.path=(\S+)", open(config_file).read()).group(1),
        "config_file": os.path.relpath(config_file, BENCH),
        "config_sha256": hashlib.sha256(open(config_file, "rb").read()).hexdigest(),
        "slurm_job_id": job_id,
        "nodes": nodes,
        "gpus_total": 8,
        "gpus_inference": 4,
        "gpus_train": n_gpus_train,
        "inference_engine": f"vllm {vllm.group(1)}" if vllm else "vllm 0.33.0",
        "train_backend": "fsdp2",
        "master_dtype": "fp32",
        "max_staleness": 4,
        "start_time": start,
        "end_time": end,
        "gpu_hours": 8 * (end - start) / 3600,
        "status": status,
        "notes": "fully async (in-flight weight sync); wait_for_rollouts_time = step - convert - train - sync; "
        "gen_tokens = avg_assistant_tokens * num_seq; trained_tokens = avg_num_tokens (completion only, no prompt; SkyRL logs no prompt length) * num_seq; "
        "forwarded_tokens_padded = batch_padded_seq_len * num_seq (remove_microbatch_padding=false pads every micro-batch "
        "to the step's max prompt + max response)",
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{len(rows)} steps -> {out}")


if __name__ == "__main__":
    main()
