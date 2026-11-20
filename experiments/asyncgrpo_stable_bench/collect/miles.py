"""miles native logs -> results/raw/<run_id>/ (collect/SCHEMA.md).

    python collect/miles.py /fsx/amine_dirhoussi/stable-bench-runs/miles/<run_id> [--status completed|failed|killed]

Reads, from the run dir written by launch/miles/w1.sbatch:
  - events/**/*.jsonl (under megatron/ when --save is set): miles' event log. Every metric dict miles logs to its
    trackers is also written there as a MetricEvent with a UTC timestamp (miles/utils/tracking_utils/tracking.py:log).
  - times.txt (JOB_START / TRAIN_START / TRAIN_END), versions.txt, config.args, gpu.csv.
  - W2 (run id miles-w2-*, launch/miles/w2.sbatch): trials/ (Harbor trial dirs written in-process by the rollout
    workers) -> episodes.jsonl (collect/agentic.py) and the W2 step fields of collect/SCHEMA.md.
miles step = rollout_id + 1 (one optimizer step per rollout id). Fields miles doesn't log are null.
Pure stdlib so it runs anywhere (uv run --no-project python collect/miles.py ...).
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime


HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from agentic import episodes_from_trials, normalize, step_fields  # noqa: E402

STEP_KEYS = ("train/step", "rollout/step", "perf/step")


def read_events(run):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "**", "events", "**", "*.jsonl"), recursive=True)):
        with open(f) as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # unterminated tail of a killed run
                if isinstance(e.get("metrics"), dict):
                    ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()
                    rows.append((ts, e.get("rollout_id"), e["metrics"]))
    return rows


def by_step(events):
    """Merge every metric dict that carries a step key into one dict per rollout id; remember the latest timestamp
    of the dict that carried `train/` metrics (end of the optimizer step)."""
    steps, t_train = {}, {}
    for ts, rid, m in events:
        sid = rid if rid is not None else next((int(m[k]) for k in STEP_KEYS if k in m), None)
        if sid is None:
            continue
        steps.setdefault(sid, {}).update(m)
        if any(k.startswith("train/") and k != "train/step" for k in m):
            t_train[sid] = max(ts, t_train.get(sid, 0.0))
    return steps, t_train


def trajectory_tokens(trials):
    """Per trial, terminus-2's per-call token counts (agent/trajectory.json `metrics`, the usage the session server
    returned): sum of prompts, sum of completions, and the last call's prompt + completion."""
    out = {}
    for f in glob.glob(os.path.join(trials, "*", "agent", "trajectory.json")):
        try:
            steps = json.load(open(f)).get("steps", [])
        except (json.JSONDecodeError, OSError):
            continue
        calls = [s["metrics"] for s in steps if s.get("source") == "agent" and s.get("metrics")]
        if calls:
            p = [c.get("prompt_tokens") or 0 for c in calls]
            c = [c.get("completion_tokens") or 0 for c in calls]
            out[f.split(os.sep)[-3]] = {"prompt_tokens_sum": sum(p), "completion_tokens": sum(c), "seq_tokens": p[-1] + c[-1]}
    return out


def w2_episodes(run, max_turns=30):
    """episodes.jsonl rows from the run's Harbor trials. An episode that used every allowed call is budget_scored."""
    trials = os.path.join(run, "trials")
    if not os.path.isdir(trials):
        return []
    tok = trajectory_tokens(trials)
    eps = []
    for r in episodes_from_trials(trials):
        r.update(tok.get(r["trial"], {}))
        r["budget_stops"] = r["n_turns"] >= max_turns
        eps.append(normalize(r))
    return sorted(eps, key=lambda e: e["end"])


def g(m, *keys):
    for k in keys:
        if k in m and m[k] is not None:
            return float(m[k])
    return None


def read_kv(path):
    """`key value` lines; a line of `k1 v1 k2 v2 ...` (versions.txt python line) is read pairwise, and a repeated
    key gets a `_version` suffix (versions.txt has `sglang <sha>` then `sglang <version>`)."""
    out = {}
    if os.path.exists(path):
        for line in open(path):
            parts = line.split()
            pairs = (
                [parts[:2]]
                if parts and parts[0] in ("JOB_START", "TRAIN_START", "TRAIN_END", "JOB_END")
                else list(zip(parts[::2], parts[1::2], strict=False))
            )
            for k, v in pairs:
                out[k if k not in out else f"{k}_version"] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="native run dir (stable-bench-runs/miles/<run_id>)")
    p.add_argument("--status", default=None, help="completed / failed / killed (default: from times.txt)")
    p.add_argument("--out", default=None, help="default: results/raw/<run_id>")
    args = p.parse_args()

    run = os.path.abspath(args.run)
    run_id = os.path.basename(run)
    out = args.out or os.path.join(BENCH, "results", "raw", run_id)
    os.makedirs(out, exist_ok=True)

    cfg = open(os.path.join(run, "config.args")).read() if os.path.exists(os.path.join(run, "config.args")) else ""
    flags = {}
    for line in cfg.splitlines():
        line = line.strip()
        if line.startswith("--"):
            k, *v = line.split(None, 1)
            flags[k] = v[0] if v else True
    n_rollouts = int(flags.get("--rollout-batch-size", 0)) * int(flags.get("--n-samples-per-prompt", 0)) or None
    max_len = int(flags["--rollout-max-response-len"]) if "--rollout-max-response-len" in flags else None

    steps, t_train = by_step(read_events(run))
    times = read_kv(os.path.join(run, "times.txt"))
    rows, prev_t = [], float(times["TRAIN_START"]) if "TRAIN_START" in times else None
    for sid in sorted(steps):
        m = steps[sid]
        if not any(k.startswith("train/") and k != "train/step" for k in m):
            continue  # rollout-only entry (e.g. the last batch generated but never trained)
        t = t_train.get(sid)
        len_mean = g(m, "rollout/response_len/mean", "rollout/response_lengths")
        zero_all = g(m, "rollout/zero_std/all_zero_percentage")
        one_all = g(m, "rollout/zero_std/all_one_percentage")
        tok_s, t_act = g(m, "perf/actor_train_tok_per_s"), g(m, "perf/actor_train_time")
        rows.append(
            {
                "step": sid + 1,
                "time": t,
                "step_time": g(m, "perf/step_time") or ((t - prev_t) if t and prev_t else None),
                # log-prob recompute + forward/backward/optimizer on the train GPUs
                "train_compute_time": g(m, "perf/train_time", "perf/actor_train_time"),
                "wait_for_rollouts_time": g(m, "perf/train_wait_time"),
                "weight_sync_time": g(m, "perf/update_weights_time"),
                "num_rollouts": n_rollouts,
                "gen_tokens": (len_mean * n_rollouts) if (len_mean is not None and n_rollouts) else None,
                # miles logs train throughput, not a token count: tok/s x actor train time (prompt + completion)
                "trained_tokens": (tok_s * t_act) if (tok_s is not None and t_act is not None) else None,
                "padding_fraction": 0.0,  # thd packing with dynamic batch: no pad tokens between samples
                "reward_mean": g(m, "rollout/raw_reward"),
                "reward_std": g(m, "rollout/raw_reward_std"),
                # groups whose 8 rewards are all 0 or all 1 (zero std -> zero advantage)
                "frac_zero_adv_groups": (zero_all + one_all)
                if (zero_all is not None and one_all is not None)
                else None,
                "completion_len_mean": len_mean,
                "completion_len_max": g(m, "rollout/response_len/max"),
                "truncated_fraction": g(m, "rollout/truncated_ratio", "rollout/truncated"),
                "staleness_mean": g(m, "rollout/fully_async/avg_staleness"),
                "staleness_max": g(m, "rollout/fully_async/max_staleness"),
                "logprob_mismatch_kl": g(m, "train/train_rollout_kl"),
                "grad_norm": g(m, "train/grad_norm"),
                "lr": g(m, "train/lr", "train/lr-pg_0"),
                "peak_mem_gb": None,
            }
        )
        prev_t = t if t else prev_t
    episodes = w2_episodes(run) if run_id.split("-")[1] == "w2" else []
    if episodes:
        with open(os.path.join(out, "episodes.jsonl"), "w") as f:
            for e in episodes:
                f.write(json.dumps(e) + "\n")
        prev = float(times["TRAIN_START"]) if "TRAIN_START" in times else 0.0
        for r in rows:
            t = r["time"] or prev
            r.update(step_fields(episodes, prev, t))
            win = [e for e in episodes if prev < e["end"] <= t and e["prompt_tokens_sum"] is not None]
            mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
            per_call = mean([e["prompt_tokens_sum"] + e["completion_tokens"] for e in win])
            own = mean([e["completion_tokens"] for e in win])
            # miles trains one sample per episode (TITO session v1: the episode's linear token sequence), so what it
            # forwards is tok/s x actor time (computed above as trained_tokens); the SHARED_W2 trained_tokens counts
            # every call's prompt + completion. Neither is logged per trained episode, so both W2 counts use the
            # episodes that finished in the step window, scaled to the step's 64 episodes.
            r["forwarded_tokens"] = r["trained_tokens"]
            r["trained_tokens"] = per_call * n_rollouts if (per_call is not None and n_rollouts) else None
            r["loss_tokens"] = own * n_rollouts if (own is not None and n_rollouts) else None
            r["prefix_share_ratio"] = (
                r["trained_tokens"] / r["forwarded_tokens"] if r["trained_tokens"] and r["forwarded_tokens"] else None
            )
            prev = t
    with open(os.path.join(out, "steps.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if os.path.exists(os.path.join(run, "gpu.csv")):
        shutil.copy(os.path.join(run, "gpu.csv"), os.path.join(out, "gpu.csv"))
    if os.path.exists(os.path.join(run, "eval.jsonl")):
        shutil.copy(os.path.join(run, "eval.jsonl"), os.path.join(out, "eval.jsonl"))

    versions = read_kv(os.path.join(run, "versions.txt"))
    start = float(times["JOB_START"]) if "JOB_START" in times else None
    end_t = times.get("TRAIN_END") or times.get("JOB_END")
    end = float(end_t) if end_t else (rows[-1]["time"] if rows else None)
    status = args.status
    if status is None:
        status = (
            "completed"
            if times.get("EXIT", None) == "0" or "EXIT 0" in open(os.path.join(run, "times.txt")).read()
            else "failed"
        )
    nodes = []
    if os.path.exists(os.path.join(run, "gpu.csv")):
        with open(os.path.join(run, "gpu.csv")) as fh:
            header = next(fh, "")
            if header.rstrip().endswith("node"):
                nodes = sorted({line.rsplit(",", 1)[-1].strip() for line in fh if line.strip()})
    cfg_rel = None
    cfg_hash = hashlib.sha256(cfg.encode()).hexdigest() if cfg else None
    for c in glob.glob(os.path.join(BENCH, "configs", "miles", "*.args")):
        if hashlib.sha256(open(c, "rb").read()).hexdigest() == cfg_hash:
            cfg_rel = os.path.relpath(c, BENCH)
    parts = run_id.split("-")
    meta = {
        "run_id": run_id,
        "lib": "miles",
        "lib_version": "v0.1.3",
        "lib_sha": versions.get("miles"),
        "trl_sha": None,
        "workload": parts[1] if len(parts) > 1 else None,
        "size": parts[2] if len(parts) > 2 else None,
        "model": {"S": "Qwen/Qwen3.5-2B", "M": "Qwen/Qwen3.5-9B"}.get(parts[2] if len(parts) > 2 else "S"),  # size letter in the run id
        "config_file": cfg_rel,
        "config_sha256": cfg_hash,
        "slurm_job_id": parts[-1],
        "nodes": nodes,
        "gpus_total": 8,
        "gpus_inference": int(flags.get("--rollout-num-gpus", 0)) or None,
        "gpus_train": int(flags.get("--actor-num-gpus-per-node", 0)) or None,
        "inference_engine": f"sglang {versions.get('sglang_version', '?')} (sglang-miles@{versions.get('sglang', '?')})",
        "train_backend": "megatron",
        "master_dtype": "fp32 (Megatron distributed optimizer main params), bf16 compute",
        "max_staleness": max((r["staleness_max"] for r in rows if r["staleness_max"] is not None), default=None),
        "start_time": start,
        "end_time": end,
        "gpu_hours": (end - start) / 3600 * 8 if (start and end) else None,
        "status": status,
        "notes": f"max_response_len {max_len}; staleness unbounded by config (buffer 2x rollout batch), "
        "max_staleness = max observed rollout/fully_async/max_staleness",
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"{run_id}: {len(rows)} steps -> {out}")


if __name__ == "__main__":
    main()
