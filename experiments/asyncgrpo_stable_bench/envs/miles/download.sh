#!/bin/bash
# Models + miles' own datasets into the shared hub cache, with plain-dir symlinks under $MD
# (miles launchers take --hf-checkpoint <dir>, and HF export copies metadata from it, so it must be a local dir).
#   sbatch --wait -p hopper-cpu --qos=low -c 8 --mem=16G -t 2:00:00 -o .../logs/download-%j.out download.sh
set -euo pipefail
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
MD=/fsx/amine_dirhoussi/stable-bench-runs/miles/models
DD=/fsx/amine_dirhoussi/stable-bench-runs/miles/datasets
mkdir -p $MD $DD
HF="uvx --from huggingface_hub[hf_xet] hf"
for m in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3-4B-Base; do
  p=$($HF download $m --quiet)
  ln -sfn "$p" $MD/$(basename $m)
done
for d in zhuzilin/dapo-math-17k zhuzilin/aime-2024; do
  p=$($HF download --repo-type dataset $d --quiet)
  ln -sfn "$p" $DD/$(basename $d)
done
ls -l $MD $DD
