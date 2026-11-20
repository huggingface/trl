#!/bin/bash
# From-scratch SkyRL install (FSDP backend + vLLM), following docs/content/docs/getting-started/installation.mdx
# ("Install without Dockerfile": uv venv outside the repo, then `uv sync --active --extra fsdp`).
# Runs on a hopper-cpu node:  sbatch -p hopper-cpu --qos=low -c 16 --mem=40G -t 2:00:00 --wait install.sh
#SBATCH --job-name=skyrl-install
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -euo pipefail

SRC=/fsx/amine_dirhoussi/stable-bench-src/skyrl
VENV=/fsx/amine_dirhoussi/envs/bench-skyrl
SHA=d87004689e3a57e4041991a3c4cf71bed333a44d   # main, 2026-11-06 (see SOURCE)

t0=$(date +%s)
[ -d "$SRC" ] || git clone https://github.com/NovaSky-AI/SkyRL.git "$SRC"
git -C "$SRC" fetch -q origin
git -C "$SRC" checkout -q "$SHA"

rm -rf "$VENV"
uv venv --python 3.12 "$VENV"
cd "$SRC"
# The cache lives on /admin and the venv on /fsx, so hardlinks are impossible: copy.
UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --extra fsdp
# The shared W1 reward (workloads/math/reward.py) needs math-verify, which SkyRL does not depend on.
UV_LINK_MODE=copy uv pip install --python "$VENV/bin/python" "math-verify[antlr4_9_3]" "antlr4-python3-runtime==4.9.3"

"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
import torch, vllm, transformers, ray, skyrl, flash_attn
import fla, causal_conv1d, math_verify, omegaconf
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
from causal_conv1d import causal_conv1d_fn  # noqa: F401
for p in ["torch", "vllm", "transformers", "ray", "flash-attn", "flash-linear-attention", "fla-core", "causal-conv1d", "flashinfer-python", "skyrl", "skyrl-gym", "math-verify"]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}: not installed")
EOF
t1=$(date +%s)
echo "INSTALL_WALL_S=$((t1 - t0))"
du -sh "$VENV"
