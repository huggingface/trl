#!/bin/bash
# prime-rl GPU smoke on 2 GPUs: import check, model/data downloads, Qwen3.5 kernel/packing check (1 GPU),
# then the reverse-text example end to end (1 inference + 1 trainer GPU) through rl.sbatch.
# Usage: b200-queue launch/prime-rl/smoke.sbatch
#SBATCH --job-name=prl-smoke
#SBATCH --partition=hopper-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out

set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
SRC=/fsx/amine_dirhoussi/stable-bench-src/prime-rl
source /fsx/amine_dirhoussi/envs/bench-prime-rl/bin/activate
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
echo "smoke_start $(date +%s.%N) $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ "${FULL:-0}" = 1 ]; then
echo "=== import check"
python - <<'EOF'
import importlib.metadata as md, time
t = time.time()
import torch, vllm, transformers, prime_rl, verifiers, fla, flash_attn, math_verify
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.conv import causal_conv1d
import math_env, reverse_text, aime24
print(f"imports ok in {time.time() - t:.1f}s; cuda {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
for p in ["prime-rl", "torch", "vllm", "transformers", "flash-linear-attention", "fla-core", "causal-conv1d",
          "flash-attn", "flash-attn-4", "verifiers", "renderers", "math-verify", "harbor", "mamba-ssm"]:
    try:
        print(" ", p, md.version(p))
    except md.PackageNotFoundError:
        print(" ", p, "NOT INSTALLED")
EOF

echo "=== downloads"
for m in Qwen/Qwen3.5-2B PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT; do
    hf download "$m" > /dev/null && echo "downloaded $m"
done
cat "$HF_HUB_CACHE"/models--Qwen--Qwen3.5-2B/snapshots/*/generation_config.json

fi

echo "=== Qwen3.5 check"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python "$BENCH/envs/prime-rl/qwen35_check.py" 2>&1 | grep -v Warning

echo "=== reverse-text rl (examples/basic/reverse-text/rl.toml, 1+1 GPUs)"
bash "$BENCH/launch/prime-rl/rl.sbatch" "$SRC/examples/basic/reverse-text/rl.toml" reverse-text-smoke \
    --monitors.wandb.project stable-bench-prime-rl
echo "smoke_end $(date +%s.%N) rc=$?"
