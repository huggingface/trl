#!/bin/bash
#SBATCH --job-name=orlhf-cc1d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/logs/%x-%j.out
# Shared torch 2.13 build, run from the OpenRLHF venv (install.sh also builds/reuses it): compile causal-conv1d 1.7.0 for sm_100 against the venv's torch 2.13+cu130 (shared with prime-rl and SkyRL), keep the wheel
# in the shared wheels dir (reusable by any venv with torch 2.13 + cu130 + py3.12), and install it.
# Runs on a GPU node because hopper-cpu nodes have neither g++ nor a CUDA toolkit:
#   b200-queue envs/openrlhf/build_causal_conv1d.sh
set -euo pipefail
VENV=/fsx/amine_dirhoussi/envs/bench-openrlhf
WHEELS=/fsx/amine_dirhoussi/stable-bench-src/wheels/torch2.13-cu130-py312
T0=$(date +%s)
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
nvcc --version | tail -2; g++ --version | head -1
"$VENV/bin/python" -c "import torch; assert torch.__version__.startswith('2.13') and torch.version.cuda.startswith('13.0'), torch.__version__"
mkdir -p "$WHEELS"
WHL=$(ls "$WHEELS"/causal_conv1d-1.7.0-*.whl 2>/dev/null | head -1 || true)
if [ -z "$WHL" ]; then
  BUILD=$(mktemp -d)
  curl -sSfL -o "$BUILD/causal_conv1d-1.7.0.tar.gz" "$(curl -sSfL https://pypi.org/pypi/causal-conv1d/1.7.0/json | "$VENV/bin/python" -c 'import json,sys; print([u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"]=="sdist"][0])')"
  # CAUSAL_CONV1D_FORCE_BUILD: compile instead of fetching a prebuilt wheel (none exists for torch 2.13).
  # --no-build-isolation: compile against the venv's torch. Only sm_100 (B200).
  export TORCH_CUDA_ARCH_LIST="10.0" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${SLURM_CPUS_ON_NODE:-11}
  uv build --wheel --no-build-isolation --python "$VENV/bin/python" --out-dir "$WHEELS" "$BUILD/causal_conv1d-1.7.0.tar.gz"
  WHL=$(ls "$WHEELS"/causal_conv1d-1.7.0-*.whl | head -1)
fi
uv pip install --python "$VENV/bin/python" --no-cache --no-deps --reinstall "$WHL"
cd /tmp && "$VENV/bin/python" -c "
import torch, causal_conv1d
from causal_conv1d import causal_conv1d_fn
x = torch.randn(2, 64, 128, device='cuda', dtype=torch.bfloat16); w = torch.randn(64, 4, device='cuda', dtype=torch.bfloat16)
print('causal_conv1d', causal_conv1d.__version__, 'ok', causal_conv1d_fn(x, w).shape)"
echo "TIMING causal_conv1d_build_s=$(( $(date +%s) - T0 ))"
ls -la "$WHEELS"; du -sh "$VENV"
