"""prime-rl native logs -> results/raw/<run_id>/ (collect/SCHEMA.md).

Reads, from `/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/`:
    <run_id>/monitors/file/metrics.jsonl   prime-rl's file monitor: trainer and orchestrator rows interleaved, each
                                           {"step", "time", "producer", ...}. Rows with step=None are time-keyed
                                           (inference / concurrency samples) and are skipped.
    <run_id>/logs/attempt_*/inference.log  only for the vLLM version line
    <run_id>-aux/gpu.csv                   1 Hz nvidia-smi from the launch script
    <run_id>-aux/job_times.txt             job_start / train_end / job_end lines from the launch script

Writes meta.json, steps.jsonl (one row per trainer step, orchestrator metrics of the same step merged in, all native
keys kept under "native") and gpu.csv. prime-rl writes no per-span timeline, so there is no spans.jsonl.

W2 runs (run_id prime-rl-w2-...) also read the file monitor's episode traces:
    <run_id>/monitors/file/traces/stream/*.jsonl[.zst]          one verifiers Episode per line (traces, per-call usage
                                                                and times, timing.{boot,setup,agent,finalize,scoring})
    <run_id>/monitors/file/traces/annotations/orchestrator.index.jsonl   which traces shipped at which step
    <run_id>-aux/host.csv                                       host memory / live containers / load every 5 s
and write episodes.jsonl plus the W2 step fields of collect/SCHEMA.md. `--traces <traces.jsonl>` instead turns a
`vf-run` output (same Episode records) into episodes.jsonl only, for sandbox tests without a training run.
Usage: python collect/prime-rl.py <run_id> [--status completed|failed|killed] [--keys]
       python collect/prime-rl.py <name> --traces <vf-run traces.jsonl>
"""

import argparse
import glob
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict


RUNS = "/fsx/amine_dirhoussi/stable-bench-runs/prime-rl"
BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "/fsx/amine_dirhoussi/stable-bench-src/prime-rl"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentic import step_fields  # noqa: E402

# stop conditions that end an episode on a budget (verified as-is, SHARED_W2 `budget_scored`)
BUDGET_STOPS = {"max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens", "max_turn_tokens", "rollout_timeout"}


def read_episodes(paths):
    """verifiers Episode records from file-monitor stream chunks (plain or zstd-sealed) or a vf-run traces.jsonl."""
    for path in paths:
        if path.endswith(".zst"):
            import zstandard

            fh = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"), read_across_frames=True)
            lines = io.TextIOWrapper(fh, encoding="utf-8")
        else:
            lines = open(path, encoding="utf-8")
        for line in lines:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:  # the live chunk's last line can be mid-write
                continue


def episode_row(ep):
    """One verifiers Episode (single agent) -> the SCHEMA.md W2 episode row, plus prime-rl's own fields."""
    tr = ep["traces"][0]
    tm = tr.get("timing") or {}
    span = lambda k: (tm.get(k) or {}).get("duration") or 0.0  # noqa: E731
    ends = [(tm.get(k) or {}).get("end") or 0.0 for k in ("boot", "setup", "agent", "finalize", "scoring")]
    start, end = tm.get("start"), max(ends)
    calls = tr.get("calls") or []
    usage = tr.get("usage") or {}
    used = [c["usage"] for c in calls if c.get("usage")]
    last = used[-1] if used else {}
    model_s = ((tm.get("agent") or {}).get("model") or {}).get("duration")
    errors = [e.get("message") for e in (tr.get("errors") or []) + (ep.get("errors") or []) if isinstance(e, dict)]
    failed = tr.get("reward") is None or tr.get("has_error") or not ep.get("ok", True) or tr.get("dropped") is not None
    budget = tr.get("stop_condition") in BUDGET_STOPS or tr.get("is_timeout") or tr.get("is_truncated")
    work = ((ep.get("run") or {}).get("work") or {})
    wall = end - start if start and end else None
    return {
        "start": start,
        "end": end,
        "wall_s": wall,
        "task": ((tr.get("task") or {}).get("data") or {}).get("name", "").split("/")[-1] or None,
        "reward": None if failed else tr.get("reward"),
        "status": "failed" if failed else ("budget_scored" if budget else "scored"),
        "error": errors[0] if errors else None,
        "turns": tr.get("num_turns"),
        "completion_tokens": usage.get("completion_tokens"),
        "seq_tokens": (last.get("prompt_tokens") or 0) + (last.get("completion_tokens") or 0) if last else None,
        "prompt_tokens_sum": usage.get("prompt_tokens"),
        # Harbor's phase names: environment_setup = box boot + task setup (upload environment/, healthcheck setup.sh)
        # + harness setup (uv installs harbor in the box); verifiers has no separate agent_setup phase.
        "phases": {
            "environment_setup": span("boot") + span("setup"),
            "agent_setup": None,
            "agent_execution": span("agent"),
            "verifier": span("finalize") + span("scoring"),
        },
        "model_s": model_s,
        "sandbox_s": wall - model_s if wall is not None and model_s is not None else None,
        "trial": tr.get("id"),
        # prime-rl / verifiers native
        "trace_id": tr.get("id"),
        "kind": work.get("type") or (ep.get("run") or {}).get("type"),
        "dispatch_step": work.get("step"),
        "policy": work.get("policy"),
        "stop_condition": tr.get("stop_condition"),
        "branches": tr.get("num_branches"),
        # SHARED_W2 trained_tokens for this episode: every model call's prompt + completion, no deduplication
        "call_tokens_sum": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0),
        "timing": {k: span(k) for k in ("boot", "setup", "agent", "finalize", "scoring")}
        | {"agent_model": model_s, "agent_harness": ((tm.get("agent") or {}).get("harness") or {}).get("duration")},
    }


def shipped_traces(index_path):
    """trace_id -> step it shipped at (the effective subset the orchestrator annotates at ship time)."""
    shipped = {}
    if os.path.exists(index_path):
        for line in open(index_path):
            r = json.loads(line)
            info = r.get("info") or {}
            if info.get("effective") and (info.get("ship") or {}).get("step") is not None:
                shipped[r["trace_id"]] = info["ship"]["step"]
    return shipped


def write_episodes(out, episodes):
    with open(os.path.join(out, "episodes.jsonl"), "w") as f:
        for e in sorted(episodes, key=lambda e: e["end"] or 0):
            f.write(json.dumps(e) + "\n")


def val(m, *keys):
    for k in keys:
        v = m.get(k)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return None


def mul(a, b):
    return None if a is None or b is None else a * b


def load_metrics(path):
    by_step = defaultdict(dict)
    time_by_step = {}
    for line in open(path):
        r = json.loads(line)
        step = r.pop("step")
        if step is None:
            continue
        producer = r.pop("producer", "unknown")
        t = r.pop("time")
        for k, v in r.items():
            by_step[step][f"{producer}:{k}"] = v
        if producer == "trainer":
            time_by_step[step] = max(time_by_step.get(step, 0.0), t)
    return by_step, time_by_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--status", default=None)
    ap.add_argument("--keys", action="store_true", help="print the native keys seen per producer and exit")
    ap.add_argument("--traces", default=None, help="a vf-run traces.jsonl: write only episodes.jsonl for <run_id>")
    args = ap.parse_args()
    if args.traces:
        out = os.path.join(BENCH, "results", "raw", args.run_id)
        os.makedirs(out, exist_ok=True)
        episodes = [episode_row(ep) for ep in read_episodes([args.traces])]
        write_episodes(out, episodes)
        print(f"{len(episodes)} episodes -> {out}/episodes.jsonl")
        return
    run_dir = os.path.join(RUNS, args.run_id)
    aux = run_dir + "-aux"
    by_step, time_by_step = load_metrics(os.path.join(run_dir, "monitors", "file", "metrics.jsonl"))

    if args.keys:
        keys = sorted({k for m in by_step.values() for k in m})
        print("\n".join(keys))
        return

    out = os.path.join(BENCH, "results", "raw", args.run_id)
    os.makedirs(out, exist_ok=True)

    rows = []
    for step in sorted(s for s in by_step if s in time_by_step):
        m = by_step[step]
        t = lambda *k: val(m, *(f"trainer:{x}" for x in k))  # noqa: E731
        o = lambda *k: val(m, *(f"orchestrator:{x}" for x in k))  # noqa: E731
        thr, fwd = t("perf/throughput"), t("time/forward_backward")
        none, all_ = o("train/agg/all/agent/solved_none"), o("train/agg/all/agent/solved_all")
        rows.append(
            {
                "step": step,
                "time": time_by_step[step],
                "step_time": t("time/step"),
                "train_compute_time": t("time/forward_backward"),
                "wait_for_rollouts_time": t("time/wait_for_batch"),
                "weight_sync_time": t("time/broadcast_weights"),
                "num_rollouts": o("progress/rollouts"),
                # Every rollout that arrived in the step's window (progress/rollouts) times their mean completion length.
                # progress/output_tokens is NOT this: it only counts the shipped (nonzero-advantage) samples.
                "gen_tokens": mul(o("progress/rollouts"), o("train/agg/all/num_output_tokens/mean")),
                # num_tokens through fwd+bwd over the step, all ranks = throughput * fwd_bwd time (trainer/rl/train.py).
                # Micro-batches are packed to their real length (pad_to_multiple_of = 1), so these are non-pad tokens of
                # the shipped samples only: zero-advantage samples are pruned before the trainer in both arms
                # (orchestrator/train_sink.py). It also counts the zero-loss dummy micro-batches prepare_batch adds to make
                # the micro-batch count divisible by the number of trainer ranks (progress/input_tokens + output_tokens
                # excludes them).
                "trained_tokens": None if thr is None or fwd is None else round(thr * fwd),
                "padding_fraction": None,
                "reward_mean": o("train/agg/all/agent/reward/mean"),
                "reward_std": None,
                # 0/1 reward: a group has zero advantage iff it is all-wrong or all-right (orchestrator/metrics.py).
                "frac_zero_adv_groups": None if none is None or all_ is None else none + all_,
                "completion_len_mean": o("train/agg/all/num_output_tokens/mean"),
                "completion_len_max": o("train/agg/all/num_output_tokens/max"),
                "truncated_fraction": o("train/agg/all/agent/is_truncated/mean"),
                "staleness_mean": o("staleness/token/mean"),
                "staleness_max": o("staleness/rollout/max"),
                "logprob_mismatch_kl": t("mismatch_kl/all/mean"),
                "grad_norm": t("optim/grad_norm"),
                "lr": t("optim/lr"),
                "peak_mem_gb": t("perf/peak_memory"),
                "native": m,
            }
        )

    w2 = "-w2-" in args.run_id
    if w2:
        traces_dir = os.path.join(run_dir, "monitors", "file", "traces")
        chunks = sorted(glob.glob(os.path.join(traces_dir, "stream", "*.jsonl*")))
        episodes = [episode_row(ep) for ep in read_episodes(chunks)]
        write_episodes(out, episodes)
        train_eps = [e for e in episodes if e["kind"] == "train" and e["end"]]
        by_trace = {e["trace_id"]: e for e in train_eps}
        shipped = shipped_traces(os.path.join(traces_dir, "annotations", "orchestrator.index.jsonl"))
        ship_by_step = defaultdict(list)
        for trace_id, step in shipped.items():
            if trace_id in by_trace:
                ship_by_step[step].append(by_trace[trace_id])
        prev = min((e["start"] for e in train_eps), default=0.0)
        for r in rows:
            ship = ship_by_step.get(r["step"], [])
            # forwarded = the trainer's own token count (thr * fwd_bwd): every shipped branch as one packed sample, its
            # whole context included, plus the zero-loss dummy micro-batches. trained = SHARED_W2's count: every model
            # call's prompt + completion of the shipped episodes, as if each call were its own sample.
            r["forwarded_tokens"] = r.pop("trained_tokens")
            r["trained_tokens"] = sum(e["call_tokens_sum"] for e in ship) if ship else None
            r["loss_tokens"] = sum(e["completion_tokens"] or 0 for e in ship) if ship else None
            r["prefix_share_ratio"] = (
                r["trained_tokens"] / r["forwarded_tokens"] if r["trained_tokens"] and r["forwarded_tokens"] else None
            )
            r["shipped_episodes"] = len(ship)
            r["shipped_branches"] = sum(e["branches"] or 0 for e in ship)
            r.update(step_fields(train_eps, prev, r["time"]))
            prev = r["time"]
    with open(os.path.join(out, "steps.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if os.path.exists(os.path.join(aux, "gpu.csv")):
        shutil.copy(os.path.join(aux, "gpu.csv"), os.path.join(out, "gpu.csv"))

    times = {}
    config_file = None
    rc = None
    for line in open(os.path.join(aux, "job_times.txt")):
        parts = line.split()
        times[parts[0]] = float(parts[1])
        for p in parts[2:]:
            if p.startswith("cfg="):
                config_file = p[4:]
            if p.startswith("rc="):
                rc = int(p[3:])
    start = times["job_start"]
    end = times.get("job_end", times.get("train_end", max(time_by_step.values(), default=start)))
    logs = sorted(glob.glob(os.path.join(run_dir, "logs", "attempt_*", "inference.log")))
    vllm = None
    if logs:
        vllm = re.search(r"vLLM (?:API server )?version v?(\d+\.\d+\.\d+)", open(logs[-1], errors="replace").read())
    sha = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    job_id = args.run_id.rsplit("-", 1)[-1]
    nodes = subprocess.run(
        ["sacct", "-nXj", job_id, "-o", "NodeList%200"], capture_output=True, text=True
    ).stdout.split()
    status = args.status or ("completed" if rc == 0 else ("failed" if rc is not None else "killed"))
    first_step = min(time_by_step.values()) - start if time_by_step else None
    parts = args.run_id.split("-")  # prime-rl-<workload>-<size>-<tag>-<job>
    resolved = os.path.join(run_dir, "configs", "latest", "resolved", "orchestrator.json")
    model = json.load(open(resolved))["model"]["name"] if os.path.exists(resolved) else "Qwen/Qwen3.5-2B"
    if w2 and os.path.exists(os.path.join(aux, "host.csv")):
        shutil.copy(os.path.join(aux, "host.csv"), os.path.join(out, "host.csv"))
    meta = {
        "run_id": args.run_id,
        "lib": "prime-rl",
        "lib_version": "0.10.0+main",
        "lib_sha": sha,
        "trl_sha": None,
        "workload": parts[2],
        "size": parts[3],
        "model": model,
        "config_file": os.path.relpath(config_file, BENCH)
        if config_file and config_file.startswith(BENCH)
        else config_file,
        "config_sha256": hashlib.sha256(open(config_file, "rb").read()).hexdigest() if config_file else None,
        "slurm_job_id": job_id,
        "nodes": nodes,
        "gpus_total": 8,
        "gpus_inference": 4,
        "gpus_train": 4,
        "inference_engine": f"vllm {vllm.group(1)}" if vllm else "vllm 0.32.0",
        "train_backend": "fsdp2",
        "attention_backend": "FA4 (flash-attn-4, attn = auto on sm_100) for full attention; fla Triton chunk_gated_delta_rule + causal_conv1d with cu_seqlens for gated DeltaNet",
        "master_dtype": "fp32",
        "max_staleness": 8,
        "start_time": start,
        "end_time": end,
        "gpu_hours": 8 * (end - start) / 3600,
        "status": status,
        "notes": f"first_step_s={first_step}: job start to the first trainer step's metrics row. "
        "Fully async: the orchestrator runs at most 1 batch ahead of the inference policy (TARGET_LAG=1), weights update in "
        "flight, tokens older than max_staleness versions are masked. step_time/train_compute_time/wait_for_rollouts_time/"
        "weight_sync_time are the trainer's time/step, time/forward_backward, time/wait_for_batch, time/broadcast_weights. "
        "staleness_mean is token-level, staleness_max is rollout-level. peak_mem_gb is perf/peak_memory: max over ranks of max_memory_reserved + memory outside the allocator, GiB. "
        "gen_tokens = progress/rollouts * train/agg/all/num_output_tokens/mean (every rollout that arrived in the step window). "
        "trained_tokens = perf/throughput * time/forward_backward, shipped (nonzero-advantage) samples only, plus any "
        "zero-loss dummy micro-batches (variable-length packed micro-batches, so no padding to "
        "seq_len; padding_fraction is not exposed). reward_mean, completion_len_*, truncated_fraction and "
        "frac_zero_adv_groups (solved_none + solved_all) cover every rollout of the step, before zero-advantage pruning. "
        "reward_std is not logged (only p10/p90).",
    }
    if w2:
        meta["sandbox_backend"] = "verifiers docker runtime on rootless podman (node-local), terminus-2 inside the box"
        meta["notes"] += (
            " W2: forwarded_tokens = perf/throughput * time/forward_backward (what the trainer ran: one packed sample per "
            "shipped branch, whole context, plus zero-loss dummy micro-batches). trained_tokens = sum over the episodes "
            "shipped at the step (orchestrator annotations, effective subset) of every model call's prompt + completion "
            "(SHARED_W2). loss_tokens = their completion tokens (stale-masked tokens not subtracted). prefix_share_ratio = "
            "trained / forwarded. Per-step episode fields cover train episodes that finished in (previous step, this step]. "
            "episodes.jsonl: model_s = verifiers' timing.agent.model (sum of model-call spans), sandbox_s = wall - model_s "
            "(box boot, task + harness setup, in-box command execution, scoring); timing.* keeps verifiers' own split."
        )
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{len(rows)} steps -> {out}; first_step_s={first_step}")


if __name__ == "__main__":
    main()
