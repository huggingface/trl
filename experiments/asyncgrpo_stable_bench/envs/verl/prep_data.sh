#!/bin/bash
# Runs after install.sh (needs the venv: verl's GSM8K script imports verl). Downloads the models and builds verl's GSM8K parquet (verl's own preprocessing script) for the smoke test and the recipe repro.
#   sbatch -p hopper-cpu --qos=low -c 8 --mem=32G -t 2:00:00 -o /fsx/amine_dirhoussi/stable-bench-runs/verl/logs/prep-%j.out envs/verl/prep_data.sh
set -euo pipefail
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
SRC=/fsx/amine_dirhoussi/stable-bench-src/verl
DATA=/fsx/amine_dirhoussi/stable-bench-runs/verl/data
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2-7B-Instruct Qwen/Qwen3.5-2B; do
  uvx --from 'huggingface_hub[hf_xet]' hf download "$m" >/dev/null && echo "downloaded $m"
done
/fsx/amine_dirhoussi/envs/bench-verl/bin/python "$SRC/examples/data_preprocess/gsm8k.py" --local_save_dir "$DATA/gsm8k"
ls -la "$DATA/gsm8k"
