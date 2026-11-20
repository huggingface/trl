#!/bin/bash
# Data prep for the SkyRL arm (hopper-cpu): math-verify, W1 data in SkyRL format, GSM8K parquet for the smoke / recipe repro, via SkyRL's own script.
#SBATCH --job-name=skyrl-prep
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -euo pipefail
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export HF_DATASETS_CACHE=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/hf-datasets
cd /fsx/amine_dirhoussi/stable-bench-src/skyrl
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
# math-verify for the shared reward (also in install.sh for fresh installs)
UV_LINK_MODE=copy uv pip install --python /fsx/amine_dirhoussi/envs/bench-skyrl/bin/python "math-verify[antlr4_9_3]" "antlr4-python3-runtime==4.9.3"
/fsx/amine_dirhoussi/envs/bench-skyrl/bin/python $BENCH/workloads/math/reward.py
/fsx/amine_dirhoussi/envs/bench-skyrl/bin/python $BENCH/configs/skyrl/w1_data.py
/fsx/amine_dirhoussi/envs/bench-skyrl/bin/python examples/train/gsm8k/gsm8k_dataset.py \
    --output_dir /fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/gsm8k
ls -la /fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/gsm8k
# Pre-download the models used on GPU so jobs don't race on the shared hub cache.
/fsx/amine_dirhoussi/envs/bench-skyrl/bin/hf download Qwen/Qwen2.5-1.5B-Instruct >/dev/null
/fsx/amine_dirhoussi/envs/bench-skyrl/bin/hf download Qwen/Qwen3.5-2B >/dev/null
echo done
