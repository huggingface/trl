#!/bin/bash
# W2 (agentic) venv for TRL: the W1 stack (same vLLM / torch / transformers / fla pins as bench-trl) plus OpenEnv's
# Harbor server and Harbor with the HF Sandboxes backend. A separate venv because vLLM pins mcp>=2 and OpenEnv's fastmcp 3
# needs mcp<2: the trainer only imports vLLM's weight-transfer code, never its MCP tool server, so mcp is overridden.
# No CUDA build here (causal-conv1d comes from the shared wheel built for bench-trl), so it runs on hopper-cpu:
#   sbatch -p hopper-cpu --qos=low -c 4 --mem=12G --requeue -t 2:00:00 -o $RUNS/install/%x-%j.out envs/trl/install_w2.sh
# The vLLM server in W2 runs from bench-trl, exactly as in W1.
set -euo pipefail
T0=$(date +%s)
VENV=${VENV:-/fsx/amine_dirhoussi/envs/bench-trl-w2}
TRL=${TRL:-/fsx/amine_dirhoussi/trl-wt-stable-bench}
OPENENV=${OPENENV:-/fsx/amine_dirhoussi/stable-bench-src/openenv}   # git checkout 86a180ede (examples/async_grpo_harbor pin)
BENCH=$TRL/experiments/asyncgrpo_stable_bench
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fsx/amine_dirhoussi/.cache/uv}
export UV_HTTP_TIMEOUT=300
PY=$VENV/bin/python
step() { echo "== [$(( $(date +%s) - T0 ))s] $*"; }

step "venv"
if { [ "${SLURM_RESTART_COUNT:-0}" = 0 ] && [ "${KEEP_VENV:-0}" = 0 ]; } || [ ! -x "$PY" ]; then
    rm -rf "$VENV"
    uv venv --python 3.12 "$VENV"
fi
[ "$(git -C "$OPENENV" rev-parse HEAD)" = 86a180ede21e044f7929b9a7783ad83aa67d83a3 ] || { echo "OpenEnv checkout is not at 86a180ede"; exit 1; }

step "vllm + trl + openenv[harbor] + harbor[hf-sandbox]"
OVERRIDES=$(mktemp)
echo 'mcp>=1.24.0,<2.0' > "$OVERRIDES"
uv pip install --python "$PY" --torch-backend=cu130 --override "$OVERRIDES" \
    "vllm==0.33.0" "torch==2.14.0" "transformers==5.20.2" \
    -e "$TRL[kernels,peft]" \
    "flash-linear-attention==0.6.0" "fla-core==0.6.0" \
    "$OPENENV[harbor]" "harbor[hf-sandbox]==0.25.0" \
    trackio datasets accelerate pandas pyarrow
rm -f "$OVERRIDES"

step "causal-conv1d (shared wheel from bench-trl's install)"
TAG=$("$PY" -c 'import sys, torch; print("torch%s-cu%s-py%d%d" % (".".join(torch.__version__.split("+")[0].split(".")[:2]), torch.version.cuda.replace(".", ""), *sys.version_info[:2]))')
WHL=$(ls /fsx/amine_dirhoussi/stable-bench-src/wheels/$TAG/causal_conv1d-1.7.0-*.whl | head -1)
uv pip install --python "$PY" --no-cache --reinstall-package causal-conv1d "$WHL"

# Harbor 0.25.0's HF sandbox upload untars as root without --no-same-owner, so every upload (task environment/, the
# verifier's tests/) fails with "Cannot change ownership". The helper has the flag; hf_sandbox.py doesn't pass it.
step "patch harbor hf_sandbox upload (tar --no-same-owner)"
SITE=$("$PY" -c 'import harbor, os; print(os.path.dirname(harbor.__file__))')
patch -N -p1 -d "$SITE/.." < "$BENCH/workloads/agentic/harbor-0.25.0-hf-sandbox-no-same-owner.patch" || grep -q 'no-same-owner' "$SITE/environments/hf_sandbox.py"

step "import check"
cd /tmp
PYTHONPATH=$OPENENV/envs "$PY" - <<'PY'
import importlib
import torch, transformers, vllm, trl, fla, causal_conv1d, mcp, harbor, openenv, litellm
from harbor_env.harness import HarborSessionFactory
from harbor.environments.hf_sandbox import HFSandboxEnvironment
from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer
from transformers.integrations.hub_kernels import resolve_internal_import
fn = resolve_internal_import(importlib.import_module("fla"), "ops.gated_delta_rule.chunk_gated_delta_rule")
assert fn is not None and fn.__module__.startswith("fla."), "fla does not import"
import causal_conv1d_cuda
import inspect
assert "--no-same-owner" in inspect.getsource(HFSandboxEnvironment.upload_dir), "harbor tar patch missing"
from importlib.metadata import version
for name in ("torch", "transformers", "vllm", "trl", "flash-linear-attention", "harbor", "openenv", "mcp", "litellm", "huggingface-hub"):
    print(f"{name:24s} {version(name)}")
PY
uv pip freeze --python "$PY" > "$BENCH/envs/trl/freeze_w2.txt"
step "done install_wall_s=$(( $(date +%s) - T0 )) venv_gb=$(du -sb "$VENV" | awk '{printf "%.1f", $1/1e9}')"
