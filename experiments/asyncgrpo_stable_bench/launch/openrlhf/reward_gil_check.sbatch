#!/bin/bash
# CPU check of the W1 reward under asyncio.to_thread (envs/openrlhf/reward_gil_check.py): original vs fixed adapter.
#SBATCH --job-name=orlhf-rewardgil
#SBATCH --partition=hopper-cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/logs/%x-%j.out
set -uo pipefail
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
source /fsx/amine_dirhoussi/envs/bench-openrlhf/bin/activate
cd /fsx/amine_dirhoussi/stable-bench-runs/openrlhf/repro
python -u $B/envs/openrlhf/reward_gil_check.py ${ADAPTERS:-$B/envs/openrlhf/w1_reward_orig.py $B/configs/openrlhf/w1_reward.py}
