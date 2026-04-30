#!/usr/bin/env python3
"""Fetch peak GPU memory from trackio system_metrics for benchmark runs.

Usage:
    python benchmark/fetch_peak_gpu_mem.py <run_name1> <run_name2> ...
    python benchmark/fetch_peak_gpu_mem.py --job-id 22092737 22092738 ...

Resolves job-id → run_name via Slurm logs (`bench-<run_name>-<job_id>.out`),
then queries `~/.cache/trackio/<project>.db:system_metrics` for max allocated
memory across all GPUs across all snapshots within the run's time window.

Output: <run_name>: peak_gb (peak_pct%) — printed one per line.
"""

import argparse
import glob
import json
import os
import sqlite3
import sys


DB_PATH = "/fsx/amine_dirhoussi/.cache/trackio/trl-sft-benchmark.db"
LOGS_DIR = "/fsx/amine_dirhoussi/trl/benchmark/logs"


def job_id_to_run_name(job_id: str) -> str | None:
    """Find the run name from the Slurm log filename pattern bench-<run_name>-<job_id>.out."""
    matches = glob.glob(os.path.join(LOGS_DIR, f"bench-*-{job_id}.out"))
    if not matches:
        return None
    fname = os.path.basename(matches[0])
    return fname.removeprefix("bench-").removesuffix(f"-{job_id}.out")


def fetch_peak_gpu_mem(run_name: str, job_id: str | None = None) -> tuple[float, float] | None:
    """Return (peak_gb, peak_pct) across all GPUs for this run, scoped to job_id's time window if provided.

    Trackio reuses run names across re-submissions, so without a time window we'd return the
    max across all historical runs with the same name. Use the Slurm log to scope to the
    actual job.
    """
    # Determine time window from Slurm log if job_id given
    started_at = None
    if job_id:
        log_files = glob.glob(os.path.join(LOGS_DIR, f"bench-*-{job_id}.out"))
        if log_files:
            with open(log_files[0]) as f:
                for line in f:
                    if line.startswith("Started: "):
                        # Format: "Started: Tue Apr 28 09:06:45 UTC 2026"
                        from datetime import datetime

                        started_at = datetime.strptime(
                            line.removeprefix("Started: ").strip(),
                            "%a %b %d %H:%M:%S %Z %Y",
                        )
                        break

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if started_at:
        # Trackio timestamps are ISO with timezone; compare lexically against the start time
        # rounded down to the second.
        start_iso = started_at.strftime("%Y-%m-%dT%H:%M:%S")
        cur.execute(
            "SELECT metrics FROM system_metrics WHERE run_name = ? AND timestamp >= ?",
            (run_name, start_iso),
        )
    else:
        cur.execute("SELECT metrics FROM system_metrics WHERE run_name = ?", (run_name,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None

    peak_gb, peak_pct, total_gb = 0.0, 0.0, 0.0
    for (m_blob,) in rows:
        m = json.loads(m_blob)
        for key in m:
            if key.startswith("gpu/") and key.endswith("/allocated_memory"):
                if m[key] > peak_gb:
                    peak_gb = m[key]
                    rank = key.split("/")[1]
                    total_gb = m.get(f"gpu/{rank}/total_memory", 80.0)
                    peak_pct = (peak_gb / total_gb * 100) if total_gb else 0
    return (peak_gb, peak_pct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="+", help="Run names or job IDs (with --job-id)")
    parser.add_argument(
        "--job-id", action="store_true", help="Treat positional args as Slurm job IDs (resolve via logs)"
    )
    args = parser.parse_args()

    for ident in args.ids:
        if args.job_id:
            run_name = job_id_to_run_name(ident)
            if run_name is None:
                print(f"{ident}: log not found", file=sys.stderr)
                continue
            result = fetch_peak_gpu_mem(run_name, job_id=ident)
            label = f"{ident} ({run_name})"
        else:
            result = fetch_peak_gpu_mem(ident)
            label = ident
        if result is None:
            print(f"{label}: no system_metrics in trackio")
        else:
            gb, pct = result
            print(f"{label}: {gb:.1f} GB ({pct:.0f}%)")


if __name__ == "__main__":
    main()
