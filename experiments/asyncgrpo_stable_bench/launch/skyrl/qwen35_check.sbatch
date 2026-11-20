#!/bin/bash
# 1-GPU Qwen3.5 checks through SkyRL's HFModelWrapper: kernel binding, packing leak, fla vs torch fwd+bwd time.
#SBATCH --job-name=skyrl-q35check
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -uo pipefail
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
source /fsx/amine_dirhoussi/envs/bench-skyrl/bin/activate
cd /fsx/amine_dirhoussi/stable-bench-src/skyrl
CHECK=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/skyrl/qwen35_check.py
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python "$CHECK"
echo "---- no fla ----"
python "$CHECK" --no-fla
