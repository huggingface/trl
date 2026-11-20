#!/bin/bash
# verl v0.9.1 with its Megatron backend (for the MoE 35B-A3B run): same checkout and lock as install.sh, extras
# vllm + megatron + math instead of vllm + fsdp. Separate venv so the FSDP runs are untouched.
#   sbatch -p hopper-cpu --qos=low -c 16 --mem=32G -t 3:00:00 -o /fsx/amine_dirhoussi/stable-bench-runs/verl/logs/install-mcore-%j.out envs/verl/install_mcore.sh
set -euo pipefail
SRC=/fsx/amine_dirhoussi/stable-bench-src/verl
VENV=/fsx/amine_dirhoussi/envs/bench-verl-mcore
T0=$(date +%s)
git -C "$SRC" log -1 --format="verl %H"
cd "$SRC"
# flash-attn: same renamed-release workaround as install.sh (step 1b).
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --all-packages --python 3.12 --extra vllm --extra megatron --extra math \
  --no-install-package flash-attn --no-install-package transformer-engine --no-install-package apex --no-install-package megatron-bridge
# The verl wheelhouse renamed its releases to <tag>-py3.12-torch2.11.0 (same builds), so the lock's URLs 404 for these
# four wheels; install the same versions from the renamed releases. megatron-bridge is installed deps-free by design (pyproject).
WH=https://github.com/verl-project/verl-wheelhouse/releases/download
uv pip install --python "$VENV/bin/python" --no-deps \
  "$WH/transformer-engine-v2.16.1-py3.12-torch2.11.0/transformer_engine-2.16.1-cp312-cp312-linux_x86_64.whl" \
  "$WH/apex-master-py3.12-torch2.11.0/apex-0.1-cp312-cp312-linux_x86_64.whl" \
  "$WH/megatron-bridge-r0.5.0-py3.12-torch2.11.0/megatron_bridge-0.5.2-py3-none-any.whl"
FA_URL=https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.11.0/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
FA_SHA=578253f946f93be01b31c89f02d6ab8d7d9d52c3d6130e9fef4ade1d8b750b4c
curl -sSfL -o /tmp/fa-$SLURM_JOB_ID.whl "$FA_URL"; echo "$FA_SHA  /tmp/fa-$SLURM_JOB_ID.whl" | sha256sum -c -
mv /tmp/fa-$SLURM_JOB_ID.whl /tmp/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
uv pip install --python "$VENV/bin/python" --no-deps /tmp/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
# causal-conv1d: the wheel envs/verl/build_causal_conv1d.sh compiled for torch 2.11 cu130 (same torch as this lock).
uv pip install --python "$VENV/bin/python" --no-deps /fsx/amine_dirhoussi/stable-bench-src/wheels/torch2.11-cu130-py312/causal_conv1d-1.7.0-*.whl
"$VENV/bin/python" - <<'PY'
import importlib
for m in ["torch", "vllm", "transformers", "verl", "fla", "megatron.core", "megatron.bridge", "mbridge", "peft"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "IMPORT FAILED", type(e).__name__, e)
PY
pip_list=$("$VENV/bin/python" -m uv pip list 2>/dev/null || uv pip list --python "$VENV/bin/python")
echo "$pip_list" | grep -i -E "transformer.engine|megatron|apex|mbridge|torch |vllm"
echo "TIMING total_s=$(( $(date +%s) - T0 ))"; du -sh "$VENV"
