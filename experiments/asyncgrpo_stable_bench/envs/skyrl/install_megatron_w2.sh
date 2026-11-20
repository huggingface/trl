#!/bin/bash
# SkyRL Megatron + Harbor venv for W2 35B (Megatron for the MoE, SkyRL's harbor extra pin for the stock HarborGenerator).
# Same checkout and lock as bench-skyrl (SOURCE), `uv sync --frozen --extra megatron` instead of fsdp.
#   sbatch -p hopper-cpu --qos=low -c 16 --mem=48G -t 3:00:00 install_megatron.sh
#SBATCH --job-name=skyrl-mcore-w2-install
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -euo pipefail
SRC=/fsx/amine_dirhoussi/stable-bench-src/skyrl
VENV=/fsx/amine_dirhoussi/envs/bench-skyrl-mcore-w2
SHA=d87004689e3a57e4041991a3c4cf71bed333a44d
echo "$(date -u) start on $(hostname); gcc: $(which g++ || echo none); nvcc: $(which nvcc || echo none)"
t0=$(date +%s)
[ "$(git -C $SRC rev-parse HEAD)" = "$SHA" ] || { echo "checkout not at $SHA"; exit 1; }
rm -rf "$VENV"
uv venv --python 3.12 "$VENV"
cd "$SRC"
UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --extra megatron --extra harbor
LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib "$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch, vllm, transformers, ray, skyrl, skyrl_gym
import megatron.core, megatron.bridge, transformer_engine
import fla, causal_conv1d, omegaconf, harbor
from harbor.trial.trial import Trial
import examples.train_integrations.harbor.entrypoints.main_harbor_fully_async
for p in ["torch", "vllm", "transformers", "ray", "megatron-core", "megatron-bridge", "transformer-engine", "transformer-engine-torch",
          "flash-attn", "flash-linear-attention", "causal-conv1d", "flashinfer-python", "skyrl", "skyrl-gym", "peft", "harbor", "litellm"]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}: not installed")
PY
echo "INSTALL_WALL_S=$(( $(date +%s) - t0 ))"
du -sh "$VENV"
