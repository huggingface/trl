# ruff: noqa: T201
#!/usr/bin/env python3
"""
Collect benchmark results from Slurm logs and/or wandb, output CSV.

Determines success/failure by grepping log files for known error patterns
(OOM, NCCL timeout, etc.) rather than relying on wandb run state.

Usage:
    # From log files (primary method)
    python benchmark/collect_results.py --logs-dir benchmark/logs

    # From wandb (supplements with metrics)
    python benchmark/collect_results.py --logs-dir benchmark/logs --wandb-project trl-sft-benchmark
"""

import argparse
import csv
import glob
import os
import re
import sys


# Error patterns to grep for in log files, ordered by priority
ERROR_PATTERNS = [
    (
        re.compile(
            r"torch\.cuda\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|out_of_memory",
            re.IGNORECASE,
        ),
        "OOM",
    ),
    (
        re.compile(
            r"NCCL.*timeout|NCCL.*watchdog|ncclSystemError|ncclInternalError",
            re.IGNORECASE,
        ),
        "NCCL timeout",
    ),
    (re.compile(r"DeepEP.*timeout|deepep.*error", re.IGNORECASE), "DeepEP timeout"),
    (
        re.compile(r"CheckpointError.*Recomputed values|CheckpointError", re.IGNORECASE),
        "MoE checkpoint error",
    ),
    (re.compile(r"RuntimeError.*CUDA|CUDA error", re.IGNORECASE), "CUDA error"),
    (re.compile(r"Killed|signal 9|SIGKILL|oom-kill", re.IGNORECASE), "OOM-killed"),
    (
        re.compile(r"ValueError|AttributeError|TypeError|KeyError", re.IGNORECASE),
        "Python error",
    ),
]

# Pattern to detect successful training completion
SUCCESS_PATTERN = re.compile(r"Training completed|train_runtime|'train_runtime'")

# Pattern to extract metrics from trainer output
METRICS_PATTERN = re.compile(
    r"'train_tokens_per_second':\s*([\d.]+)|"
    r"'mfu':\s*([\d.]+)|"
    r"'train_runtime':\s*([\d.]+)"
)


def parse_run_name(run_name: str) -> dict:
    """Parse run_id like 'bench-qwen3_4b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1-12345' into fields."""
    result = {}
    for prefix in ("dp", "tp", "pp", "cp", "ep"):
        match = re.search(rf"_{prefix}(\d+)", run_name)
        if match:
            result[prefix] = int(match.group(1))
    ctx_match = re.search(r"_ctx(\d+)k", run_name)
    if ctx_match:
        result["context_length_k"] = int(ctx_match.group(1))
    nodes_match = re.search(r"_n(\d+)_", run_name)
    if nodes_match:
        result["nodes"] = int(nodes_match.group(1))
    for backend in ("fsdp2", "deepspeed_zero3", "deepspeed"):
        if backend in run_name:
            result["backend"] = backend
            break
    # Extract model name
    model_match = re.search(r"bench-(.*?)_ctx", run_name)
    if model_match:
        result["model"] = model_match.group(1)
    return result


def diagnose_log_files(out_path: str, err_path: str) -> tuple[str, dict]:
    """
    Read log files and determine success/failure + extract metrics.

    Returns:
        (status_string, metrics_dict)
    """
    content = ""
    if os.path.exists(out_path):
        with open(out_path, errors="replace") as f:
            content += f.read()
    if os.path.exists(err_path):
        with open(err_path, errors="replace") as f:
            content += f.read()

    if not content:
        return "No (empty logs)", {}

    # Detect gradient checkpointing from the generated sbatch file
    gc_enabled = False
    run_id_match = re.search(r"bench-(.+)-\d+", os.path.basename(out_path))
    if run_id_match:
        generated_dir = os.path.join(
            os.path.dirname(os.path.dirname(out_path)),
            "generated",
            run_id_match.group(1),
        )
        sbatch_path = os.path.join(generated_dir, "job.sbatch")
        if os.path.exists(sbatch_path):
            with open(sbatch_path) as f:
                gc_enabled = "--gradient_checkpointing true" in f.read()

    # Check for success FIRST — stderr may contain NCCL warnings even on successful runs
    if SUCCESS_PATTERN.search(content):
        # Extract metrics
        metrics = {}
        for match in METRICS_PATTERN.finditer(content):
            if match.group(1):
                metrics["tps"] = float(match.group(1))
            if match.group(2):
                metrics["mfu"] = float(match.group(2))
            if match.group(3):
                metrics["runtime"] = float(match.group(3))
        metrics["gradient_checkpointing"] = gc_enabled
        return "Yes", metrics

    # Check for error patterns (only if not successful)
    for pattern, label in ERROR_PATTERNS:
        if pattern.search(content):
            return f"No ({label})", {"gradient_checkpointing": gc_enabled}

    return "No (unknown failure)", {"gradient_checkpointing": gc_enabled}


def collect_from_logs(logs_dir: str) -> list[dict]:
    """Collect results by parsing Slurm log files. Only keeps the latest job per run config."""
    # Find all .out files
    out_files = glob.glob(os.path.join(logs_dir, "bench-*.out"))
    if not out_files:
        out_files = glob.glob(os.path.join(logs_dir, "*.out"))

    # Group by run_id (everything before the last -<jobid>.out) and keep the best run:
    # prefer successful > OOM > other failures, then highest job_id as tiebreaker
    runs = {}
    for out_path in out_files:
        basename = os.path.basename(out_path)
        parts = basename.rsplit("-", 1)
        if len(parts) == 2:
            run_id = parts[0]
            job_id = int(parts[1].replace(".out", ""))
        else:
            run_id = basename
            job_id = 0
        err_path = out_path.replace(".out", ".err")
        status, _ = diagnose_log_files(out_path, err_path)
        # Priority: success=2, OOM=1, other=0
        priority = 2 if status == "Yes" else (1 if "OOM" in status else 0)
        prev = runs.get(run_id)
        if prev is None or (priority, job_id) > (prev[1], prev[2]):
            runs[run_id] = (out_path, priority, job_id)

    results = []
    for out_path, _, _ in sorted(runs.values(), key=lambda x: x[0]):
        basename = os.path.basename(out_path)
        err_path = out_path.replace(".out", ".err")

        parsed = parse_run_name(basename)
        status, metrics = diagnose_log_files(out_path, err_path)

        num_gpus = parsed.get("nodes", 1) * 8
        tps = metrics.get("tps")
        mfu = metrics.get("mfu")
        tps_per_gpu = round(tps / num_gpus, 2) if tps else None

        gc = metrics.get("gradient_checkpointing", False)

        results.append(
            {
                "Model": parsed.get("model", basename),
                "Context length (k tokens)": parsed.get("context_length_k", ""),
                "Nodes": parsed.get("nodes", ""),
                "Distributed backend": parsed.get("backend", ""),
                "CPU Offload": "false",
                "Activation Checkpointing": str(gc).lower(),
                "DP": parsed.get("dp", ""),
                "TP": parsed.get("tp", ""),
                "PP": parsed.get("pp", ""),
                "CP": parsed.get("cp", ""),
                "EP": parsed.get("ep", ""),
                "MFU": round(mfu, 2) if mfu else "-",
                "TPS": round(tps, 2) if tps else "-",
                "TPS/GPU": tps_per_gpu if tps_per_gpu else "-",
                "wandb": "on",
                "Success?": status,
            }
        )

    return results


def collect_from_wandb(project: str, entity: str | None = None, logs_dir: str | None = None) -> list[dict]:
    """Collect results from wandb, optionally cross-referencing with log files for failure diagnosis."""
    import wandb as wb

    api = wb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)

    results = []
    for run in runs:
        parsed = parse_run_name(run.name)
        summary = run.summary

        tps = summary.get("train_tokens_per_second")
        mfu = summary.get("mfu")
        num_gpus = parsed.get("nodes", 1) * 8
        tps_per_gpu = round(tps / num_gpus, 2) if tps else None

        # Determine success: check log files if available, fall back to wandb state
        status = "Yes" if run.state == "finished" else f"No ({run.state})"
        if logs_dir:
            # Find matching log file
            pattern = os.path.join(logs_dir, f"*{run.name}*.out")
            matches = glob.glob(pattern)
            if matches:
                err_path = matches[0].replace(".out", ".err")
                log_status, log_metrics = diagnose_log_files(matches[0], err_path)
                status = log_status
                # Use log metrics if wandb doesn't have them
                if not tps and log_metrics.get("tps"):
                    tps = log_metrics["tps"]
                    tps_per_gpu = round(tps / num_gpus, 2)
                if not mfu and log_metrics.get("mfu"):
                    mfu = log_metrics["mfu"]

        results.append(
            {
                "Model": run.config.get("model_name_or_path", parsed.get("model", run.name)),
                "Context length (k tokens)": parsed.get("context_length_k", ""),
                "Nodes": parsed.get("nodes", ""),
                "Distributed backend": parsed.get("backend", ""),
                "CPU Offload": str(run.config.get("fsdp_offload_params", False)).lower(),
                "DP": parsed.get("dp", ""),
                "TP": parsed.get("tp", ""),
                "PP": parsed.get("pp", ""),
                "CP": parsed.get("cp", ""),
                "EP": parsed.get("ep", ""),
                "MFU": round(mfu, 2) if mfu else "-",
                "TPS": round(tps, 2) if tps else "-",
                "TPS/GPU": tps_per_gpu if tps_per_gpu else "-",
                "wandb": "on" if run.url else "off",
                "Success?": status,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect TRL SFT benchmark results")
    parser.add_argument("--logs-dir", help="Directory with Slurm log files (primary method)")
    parser.add_argument("--wandb-project", help="Wandb project name (optional, supplements log parsing)")
    parser.add_argument("--wandb-entity", help="Wandb entity (team or user)")
    parser.add_argument("--output", "-o", help="Output CSV file (default: stdout)")
    args = parser.parse_args()

    if not args.logs_dir and not args.wandb_project:
        parser.error("At least one of --logs-dir or --wandb-project is required")

    if args.logs_dir and not args.wandb_project:
        results = collect_from_logs(args.logs_dir)
    elif args.wandb_project:
        results = collect_from_wandb(args.wandb_project, args.wandb_entity, args.logs_dir)
    else:
        results = []

    if not results:
        print("No runs found.", file=sys.stderr)
        return

    fieldnames = [
        "Model",
        "Context length (k tokens)",
        "Nodes",
        "Distributed backend",
        "CPU Offload",
        "Activation Checkpointing",
        "DP",
        "TP",
        "PP",
        "CP",
        "EP",
        "MFU",
        "TPS",
        "TPS/GPU",
        "wandb",
        "Success?",
    ]

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

    if args.output:
        out.close()
        print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
