#!/bin/bash
#SBATCH --job-name=verl-q35check
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/%x-%j.out
# Qwen3.5 kernel/packing check on 1 GPU: b200-queue launch/verl/qwen35_check.sbatch
set -uo pipefail
source /fsx/amine_dirhoussi/envs/bench-verl/bin/activate
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
cd /fsx/amine_dirhoussi/stable-bench-src/verl
python /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/verl/qwen35_check.py Qwen/Qwen3.5-2B
