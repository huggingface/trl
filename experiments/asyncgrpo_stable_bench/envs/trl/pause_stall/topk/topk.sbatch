#!/bin/bash
#SBATCH -J topk-sc
#SBATCH -p hopper-dev
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/pause_stall/topk/topk-%j.out
set -uo pipefail
export PATH=/fsx/amine_dirhoussi/envs/bench-trl/bin:$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/pause_stall/topk
python gen.py /fsx/amine_dirhoussi/stable-bench-runs/trl/trl-w1-S-recommended-364705/checkpoint-200 gen200.pt > gen-$SLURM_JOB_ID.log 2>&1 || { grep -iE "error|cuda|kernel" gen-$SLURM_JOB_ID.log | grep -v Warning | head -30; exit 1; }
python gen.py /fsx/amine_dirhoussi/stable-bench-runs/trl/trl-w1-S-recommended-364705/checkpoint-150 gen150.pt > gen-$SLURM_JOB_ID.log 2>&1 || { grep -iE "error|cuda|kernel" gen-$SLURM_JOB_ID.log | grep -v Warning | head -30; exit 1; }
python score.py /fsx/amine_dirhoussi/stable-bench-runs/trl/trl-w1-S-recommended-364705/checkpoint-200 gen200.pt
python score.py /fsx/amine_dirhoussi/stable-bench-runs/trl/trl-w1-S-recommended-364705/checkpoint-200 gen150.pt
