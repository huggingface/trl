#!/usr/bin/env python3
"""
Parse [LOAD-T] markers from bench-*-<JOB>.err logs and produce a per-rank stage-by-stage timing
breakdown for the I3 slow-loading investigation.

Usage:
    python benchmark/analyze_load_timing.py 22104071 22104072 ...
"""

import os
import re
import sys
from collections import defaultdict


MARKER = re.compile(
    r"\[LOAD-T\]\s+rank=(\d+)(?:\s+t=\s*([\d.]+)s)?\s+stage=(\w+)(?:\s+elapsed=\s*([\d.]+)s)?(?:\s+t_wall=\s*([\d.]+))?"
)


def parse_log(path):
    """Yield (rank, stage, t_rel, t_wall, elapsed_within) per marker line."""
    with open(path, errors="replace") as f:
        for line in f:
            m = MARKER.search(line)
            if not m:
                continue
            rank = int(m.group(1))
            t_rel = float(m.group(2)) if m.group(2) else None
            stage = m.group(3)
            elapsed_within = float(m.group(4)) if m.group(4) else None
            t_wall = float(m.group(5)) if m.group(5) else None
            yield rank, stage, t_rel, t_wall, elapsed_within


def report_job(job_id):
    log_dir = "/fsx/amine_dirhoussi/trl/benchmark/logs"
    files = [f for f in os.listdir(log_dir) if f.endswith(f"-{job_id}.out") or f.endswith(f"-{job_id}.err")]
    if not files:
        print(f"=== {job_id}: no log files found ===")
        return
    paths = [os.path.join(log_dir, f) for f in files]

    # Use rank 0 for the canonical timeline; collect inner-elapsed markers as max-across-ranks.
    timeline = {}  # rank0 stage -> t_rel
    inner = {}  # stage -> max elapsed
    wall_first = None
    rank0_first_step_wall = None

    rank_seen = defaultdict(dict)  # rank -> stage -> t_rel
    rank_wall = defaultdict(dict)  # rank -> stage -> t_wall
    for path in paths:
        for rank, stage, t_rel, t_wall, ew in parse_log(path):
            if t_rel is not None and stage not in rank_seen[rank]:
                rank_seen[rank][stage] = t_rel
            if t_wall is not None and stage not in rank_wall[rank]:
                rank_wall[rank][stage] = t_wall
            if ew is not None:
                inner[stage] = max(inner.get(stage, 0), ew)

    # Use rank 0 if available, else any rank
    canonical_rank = 0 if 0 in rank_seen else min(rank_seen.keys()) if rank_seen else None
    if canonical_rank is None:
        print(f"\n=== {job_id}: no markers ===")
        return
    timeline = rank_seen[canonical_rank]
    if "first_training_step_entry" in rank_wall.get(canonical_rank, {}):
        rank0_first_step_wall = rank_wall[canonical_rank]["first_training_step_entry"]

    print(f"\n=== {job_id} (canonical rank={canonical_rank}) ===")
    print(f"  logs: {[os.path.basename(p) for p in paths]}")
    if timeline:
        print("  rank-0 timeline (s from main_entry):")
        order = [
            "main_entry",
            "kernel_warm_done",
            "model_string_assigned_for_EP",
            "from_pretrained_start",
            "from_pretrained_done",
            "dataset_load_start",
            "dataset_load_done",
            "trainer_init_start",
            "trainer_init_done",
            "trainer_train_start",
            "_get_dataloader_first_call",
            "first_training_step_entry",
        ]
        prev = 0.0
        for s in order:
            if s in timeline:
                t = timeline[s]
                print(f"    {s:<40} {t:7.2f}s   (+{t - prev:6.2f}s)")
                prev = t
    if inner:
        print("  inner stages (max-across-ranks elapsed):")
        for s, e in inner.items():
            print(f"    {s:<40} {e:7.2f}s")
    if rank0_first_step_wall and timeline.get("main_entry") is not None:
        # Total time-to-first-step from rank-0's perspective = first_training_step_entry t_wall - main_entry t_wall
        # We don't have main_entry t_wall directly, so estimate via t_rel
        # If we know rank0_first_step_wall, and t_rel(main_entry) = 0 from t_wall(main_entry),
        # then time-to-first-step ≈ t_wall(first_step) - (t_wall(main_entry))
        # Easier: just look at first_training_step_entry t_wall if rank0 logged it
        print(f"  rank-0 first_training_step_entry t_wall: {rank0_first_step_wall}")


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_load_timing.py <JOB_ID> [<JOB_ID> ...]")
        sys.exit(1)
    for j in sys.argv[1:]:
        report_job(j)


if __name__ == "__main__":
    main()
