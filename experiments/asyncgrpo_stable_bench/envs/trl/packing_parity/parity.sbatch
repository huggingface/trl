#!/bin/bash
#SBATCH -J trl-parity
#SBATCH --nodes=1
#SBATCH -t 01:00:00
#SBATCH -o /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/packing_parity/parity-%j.out
set -uo pipefail
VENV=/fsx/amine_dirhoussi/envs/bench-trl
export PATH=$VENV/bin:$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONPATH=/fsx/amine_dirhoussi/trl-wt-stable-bench PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1
nvidia-smi --query-gpu=name --format=csv,noheader
python /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/packing_parity/trl_packing_parity.py
