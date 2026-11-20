#!/bin/bash
#SBATCH --job-name=robust-scan
#SBATCH --partition=hopper-cpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/results/robustness/logs/%x-%j.out

# Rerunnable and idempotent: every output is written to a .tmp file and renamed, so a spot
# preemption + requeue just redoes the scan. MODE=summary skips the log scan.
#   sbatch --wait scan_logs.sbatch                    # scan + summary
#   sbatch --wait --export=ALL,MODE=summary scan_logs.sbatch
set -euo pipefail
DIR=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/results/robustness
cd "$DIR"
echo "start $(date -u +%FT%TZ) host=$(hostname) restart=${SLURM_RESTART_COUNT:-0} mode=${MODE:-all}"
if [ "${MODE:-all}" != "summary" ]; then
  uv run --no-project --python 3.12 python "$DIR/scan_logs.py" --procs "${SLURM_CPUS_PER_TASK:-4}"
fi
uv run --no-project --python 3.12 python "$DIR/summary.py"
echo "done $(date -u +%FT%TZ)"
