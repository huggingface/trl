#!/bin/bash
# Qwen3.5 kernel binding / packing boundary / speed check on OpenRLHF's Actor class (envs/openrlhf/qwen35_check.py).
#   b200-queue --gres=gpu:1 launch/openrlhf/qwen35_check.sbatch
#SBATCH --job-name=orlhf-q35
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/logs/%x-%j.out
set -uo pipefail
source /fsx/amine_dirhoussi/envs/bench-openrlhf/bin/activate
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
cd /tmp && python $B/envs/openrlhf/qwen35_check.py "$(ls -d $HF_HUB_CACHE/models--Qwen--Qwen3.5-2B/snapshots/*/ | head -1)"
