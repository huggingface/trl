#!/bin/bash
# W2 venv for SkyRL's own recommended recipe on local Apptainer: `uv sync --frozen --extra fsdp --extra harbor`, i.e. the
# lock as shipped, including SkyRL's Harbor pin (git 3de07a0e), which has the singularity environment. No patches.
#   sbatch -p hopper-cpu --qos=low -c 8 --mem=24G --requeue -t 2:00:00 -o /fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out envs/skyrl/install_w2_local.sh
#SBATCH --job-name=skyrl-install-w2local
set -euo pipefail
T0=$(date +%s)
SRC=/fsx/amine_dirhoussi/stable-bench-src/skyrl
VENV=${VENV:-/fsx/amine_dirhoussi/envs/bench-skyrl-w2local}
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
export UV_HTTP_TIMEOUT=300
PY=$VENV/bin/python
[ "$(git -C "$SRC" rev-parse HEAD)" = d87004689e3a57e4041991a3c4cf71bed333a44d ] || { echo "wrong SkyRL rev"; exit 1; }
rm -rf "$VENV"; uv venv --python 3.12 "$VENV"
cd "$SRC"
UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --extra fsdp --extra harbor
"$PY" - <<'PY'
from importlib.metadata import version
import torch, vllm, transformers, ray, skyrl, flash_attn, fla, causal_conv1d, omegaconf, litellm  # noqa: F401
from harbor.environments.singularity.singularity import SingularityEnvironment  # noqa: F401
from examples.train_integrations.harbor.entrypoints.main_harbor_fully_async import HarborFullyAsyncExp  # noqa: F401
for p in ["torch", "vllm", "transformers", "ray", "flash-linear-attention", "skyrl", "harbor", "litellm"]:
    print(f"{p}=={version(p)}")
PY
uv pip freeze --python "$PY" > "$BENCH/envs/skyrl/freeze_w2local.txt"
echo "done install_wall_s=$(( $(date +%s) - T0 ))"
