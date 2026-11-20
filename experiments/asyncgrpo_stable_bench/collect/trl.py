"""
TRL run dir -> results/raw/<run_id>/ in the schema of collect/SCHEMA.md.

    python collect/trl.py /fsx/amine_dirhoussi/stable-bench-runs/trl/<run_id> [--status completed|failed|killed]

Reads what launch/trl/w1.sbatch and the BenchCallback / W1RolloutLoop of configs/trl/w1-*.py write into the run dir:
metrics.jsonl (every trainer log dict), spans-train.jsonl, spans-generate-*.jsonl, mem-rank*.jsonl, gpu.csv, job_start,
job_end, nodes, trl_sha, freeze.txt, launch.env, and eval.jsonl if workloads/math/eval.py was pointed there.
W2 (agentic) runs are recognised by the `split_tasks.jsonl` launch/trl/w2.sbatch copies into the run dir; they also
get episodes.jsonl (from the run dir's episodes.jsonl, see collect/agentic.py) and the W2 step fields of SCHEMA.md.
Only the standard library is used, so any python runs it.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil

from agentic import normalize, read_episode_records, step_fields


BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_COMPLETION = 8192


def read_jsonl(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def read(path, default=None):
    return open(path).read().strip() if os.path.exists(path) else default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--status", default=None)
    args = p.parse_args()
    run_dir = args.run_dir.rstrip("/")
    run_id = os.path.basename(run_dir)
    out = os.path.join(BENCH, "results", "raw", run_id)
    os.makedirs(out, exist_ok=True)

    env = dict(line.split("=", 1) for line in (read(f"{run_dir}/launch.env", "") or "").splitlines() if "=" in line)
    workload = "w2" if os.path.exists(f"{run_dir}/split_tasks.jsonl") else "w1"
    size = env.get("SIZE", "S" if workload == "w1" else "M")
    nvllm, ntrain = int(env.get("NVLLM", 4)), int(env.get("NTRAIN", 4))
    max_steps = int(env.get("MAX_STEPS", 200 if workload == "w1" else 60))
    episodes = sorted(
        (normalize(r) for r in read_episode_records(f"{run_dir}/episodes.jsonl")), key=lambda e: e["end"]
    )

    # metrics.jsonl holds one row per optimizer step (logging_steps=1) plus the final train summary row.
    logs = [r for r in read_jsonl(f"{run_dir}/metrics.jsonl") if "train_runtime" not in r]
    logs = list({r["step"]: r for r in logs}.values())  # a resumed run re-logs nothing before its checkpoint
    mem = read_jsonl(f"{run_dir}/mem-rank*.jsonl")
    gen = read_jsonl(f"{run_dir}/spans-generate-*.jsonl")
    train_spans = read_jsonl(f"{run_dir}/spans-train.jsonl")

    peak_by_step = {}
    for m in mem:
        peak_by_step[m["step"]] = max(peak_by_step.get(m["step"], 0.0), m["peak_reserved_gb"])

    steps, prev_time = [], float(read(f"{run_dir}/job_start", "0"))
    gen_sorted = sorted(gen, key=lambda s: s["end"])
    gi = 0
    for r in sorted(logs, key=lambda r: r["step"]):
        # gen_tokens: completion tokens of the sampling requests that finished during this step's window
        window = []
        while gi < len(gen_sorted) and gen_sorted[gi]["end"] <= r["time"]:
            if gen_sorted[gi]["end"] > prev_time:
                window.append(gen_sorted[gi])
            gi += 1
        fwd_bwd, opt = r.get("perf/fwd_bwd_s"), r.get("perf/optimizer_s")
        steps.append(
            {
                "step": r["step"],
                "time": r["time"],
                "step_time": r.get("perf/step_s"),
                "train_compute_time": fwd_bwd + opt if fwd_bwd is not None and opt is not None else None,
                "wait_for_rollouts_time": r.get("perf/rollout_wait_s"),
                "weight_sync_time": r.get("perf/weight_sync_s"),
                "num_rollouts": r.get("batch/samples_per_step"),
                "gen_tokens": sum(s["tokens"] for s in window) if gen else None,
                "trained_tokens": r.get("batch/trained_tokens_per_step"),
                "padding_fraction": r.get("batch/pad_frac"),
                "reward_mean": r.get("reward"),
                "reward_std": r.get("reward_group_std"),
                "frac_zero_adv_groups": r.get("frac_reward_zero_std"),
                "completion_len_mean": r.get("completions/mean_length"),
                "completion_len_max": r.get("completions/max_length"),
                "truncated_fraction": r.get("completions/clipped_ratio"),
                "staleness_mean": r.get("sample/staleness_mean"),
                "staleness_max": r.get("sample/staleness_max"),
                "logprob_mismatch_kl": None,  # TRL logs |logp_trainer - logp_vllm| (see native), not a KL
                "grad_norm": r.get("grad_norm"),
                "lr": r.get("learning_rate"),
                "peak_mem_gb": peak_by_step.get(r["step"]),
                "native": r,
            }
        )
        if workload == "w2":
            # SHARED_W2 token counts: TRL's batch/trained_tokens_per_step is the loss-mask count, and
            # batch/forwarded_tokens_per_step the unique tokens after tree packing; batch/packing_ratio is
            # (sum of per-sample lengths) / (packed row tokens), so their product is the no-dedup per-rollout count.
            fwd, ratio = r.get("batch/forwarded_tokens_per_step"), r.get("batch/packing_ratio")
            steps[-1].update(
                trained_tokens=fwd * ratio if fwd is not None and ratio is not None else None,
                loss_tokens=r.get("batch/trained_tokens_per_step"),
                forwarded_tokens=fwd,
                prefix_share_ratio=ratio,
                **step_fields(episodes, prev_time, r["time"]),
            )
        prev_time = r["time"]
    with open(f"{out}/steps.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in steps)

    spans = [{**s, "gpu": None} for s in train_spans]
    spans += [
        {"kind": "generate", "worker": s["worker"], "gpu": None, "start": s["start"], "end": s["end"],
         "step": s["version"], "prompt_tokens": s["prompt_tokens"], "tokens": s["tokens"]}
        for s in gen
    ]  # fmt: skip
    spans.sort(key=lambda s: s["start"])
    with open(f"{out}/spans.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in spans)
    with open(f"{out}/mem.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in sorted(mem, key=lambda m: (m["step"], m["rank"])))
    if workload == "w2":
        with open(f"{out}/episodes.jsonl", "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in episodes)
        spans += [
            {"kind": "env", "worker": "harbor", "gpu": None, "start": e["start"], "end": e["end"], "step": None,
             "task": e["task"], "status": e["status"]}
            for e in episodes
        ]  # fmt: skip
        spans.sort(key=lambda s: s["start"])
        with open(f"{out}/spans.jsonl", "w") as f:
            f.writelines(json.dumps(s) + "\n" for s in spans)
    for name in ("gpu.csv", "eval.jsonl"):
        if os.path.exists(f"{run_dir}/{name}"):
            shutil.copy(f"{run_dir}/{name}", f"{out}/{name}")

    freeze = read(f"{run_dir}/freeze.txt", "")
    version = lambda pkg: (re.search(rf"^{pkg}(?:==| @ )(\S+)", freeze, re.M) or [None, None])[1]  # noqa: E731
    config = f"{run_dir}/{workload}-{size}.py"
    start, end = float(read(f"{run_dir}/job_start", "nan")), read(f"{run_dir}/job_end")
    end = float(end) if end else (steps[-1]["time"] if steps else None)
    last_step = steps[-1]["step"] if steps else 0
    status = args.status or ("completed" if read(f"{run_dir}/job_end") and last_step >= max_steps else "failed")
    gpus = 8 if nvllm + ntrain == 8 else nvllm + ntrain  # timed runs hold the whole node
    meta = {
        "run_id": run_id,
        "lib": "trl",
        "lib_version": version("trl"),
        "lib_sha": read(f"{run_dir}/trl_sha"),
        "trl_sha": read(f"{run_dir}/trl_sha"),
        "workload": workload,
        "size": size,
        "model": {"S": "Qwen/Qwen3.5-2B", "M": "Qwen/Qwen3.5-9B", "L": "Qwen/Qwen3.8-27B"}[size],
        "config_file": f"configs/trl/{workload}-{size}.py",
        "config_sha256": hashlib.sha256(open(config, "rb").read()).hexdigest() if os.path.exists(config) else None,
        "slurm_job_id": run_id.rsplit("-", 1)[-1],
        "nodes": (read(f"{run_dir}/nodes", "") or "").split(),
        "gpus_total": gpus,
        "gpus_inference": nvllm,
        "gpus_train": ntrain,
        "inference_engine": f"vllm {version('vllm')}",
        "train_backend": "fsdp2",
        "master_dtype": "float32",
        "max_staleness": 4,
        "start_time": start,
        "end_time": end,
        "gpu_hours": (end - start) / 3600 * gpus if end else None,
        "status": status,
        "notes": f"launch env: {env}; trl diff vs sha: {os.path.getsize(f'{run_dir}/trl.diff') if os.path.exists(f'{run_dir}/trl.diff') else 'n/a'} bytes",
        "first_step_s": steps[0]["time"] - start if steps else None,
    }
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("run_id", "status", "first_step_s", "gpu_hours")}))
    print(f"{len(steps)} steps, {len(spans)} spans, {len(mem)} mem rows -> {out}")


if __name__ == "__main__":
    main()
