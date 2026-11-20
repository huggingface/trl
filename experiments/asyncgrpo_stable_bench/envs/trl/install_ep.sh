#!/bin/bash
# Venv for the AsyncGRPO MoE expert-parallel work: envs/trl/install.sh with TRL from the trl-wt-asyncgrpo-ep worktree
# and transformers >= 5.20.2 (DistributedConfig `ep_size`, released). Runs on hopper-cpu: the causal-conv1d wheel for
# torch2.14-cu130-py312 is already in the shared wheel dir.
# The trl-wt-asyncgrpo-ep worktree carried the expert-parallel fixes that later landed on main (#8236, #8294, #8295),
# so TRL main from after those PRs replaces it.
#   sbatch -p hopper-cpu --qos=low -c 8 --mem=24G --requeue -t 2:00:00 \
#     -o /fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/install/%x-%j.out envs/trl/install_ep.sh
# Also stages the models in the shared hub cache (DOWNLOAD=0 skips that).
set -euo pipefail
T0=$(date +%s)
VENV=${VENV:-/fsx/amine_dirhoussi/envs/bench-trl-ep}
TRL=${TRL:-/fsx/amine_dirhoussi/trl-wt-asyncgrpo-ep}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fsx/amine_dirhoussi/.cache/uv}
export UV_HTTP_TIMEOUT=300 HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
PY=$VENV/bin/python
step() { echo "== [$(( $(date +%s) - T0 ))s] $*"; }

step "venv"
if { [ "${SLURM_RESTART_COUNT:-0}" = 0 ] && [ "${KEEP_VENV:-0}" = 0 ]; } || [ ! -x "$PY" ]; then
    rm -rf "$VENV"
    uv venv --python 3.12 "$VENV"
fi

step "vllm + trl + deps"
uv pip install --python "$PY" --torch-backend=cu130 \
    "vllm>=0.32,<0.34" \
    -e "$TRL[kernels,peft,math_verify,test]" \
    "transformers>=5.20.2" \
    "flash-linear-attention" \
    "math-verify[antlr4_13_2]>=0.8.0" \
    trackio datasets accelerate ninja packaging wheel setuptools

step "flashinfer precompiled kernels"
FI=$(uv pip show --python "$PY" flashinfer-python | sed -n 's/^Version: //p')
uv pip install --python "$PY" --no-deps "flashinfer-jit-cache==$FI" \
    --extra-index-url https://flashinfer.ai/whl/cu130 --index-strategy unsafe-best-match

step "causal-conv1d"
TAG=$("$PY" -c 'import sys, torch; print("torch%s-cu%s-py%d%d" % (".".join(torch.__version__.split("+")[0].split(".")[:2]), torch.version.cuda.replace(".", ""), *sys.version_info[:2]))')
WHL=$(ls /fsx/amine_dirhoussi/stable-bench-src/wheels/$TAG/causal_conv1d-1.7.0-*.whl | head -1)
echo "causal-conv1d wheel: $WHL"
uv pip install --python "$PY" --no-cache --reinstall-package causal-conv1d "$WHL"

step "import check"
cd /tmp
"$PY" - <<'PY'
import importlib
import torch, transformers, vllm, trl, fla, causal_conv1d, kernels, accelerate
from transformers.distributed import DistributedConfig
from transformers.integrations.hub_kernels import resolve_internal_import
fn = resolve_internal_import(importlib.import_module("fla"), "ops.gated_delta_rule.chunk_gated_delta_rule")
assert fn is not None and fn.__module__.startswith("fla."), "fla does not import"
print("ep_size ok:", DistributedConfig(fsdp_size=8, ep_size=8).efsdp_size)
for m in (torch, transformers, vllm, trl, fla, causal_conv1d, kernels, accelerate):
    print(f"{m.__name__:16s} {getattr(m, '__version__', '?')}")
print("trl from", trl.__file__)
PY
uv pip freeze --python "$PY" > /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/freeze_ep.txt || true

if [ "${DOWNLOAD:-1}" = 1 ]; then
    for m in Qwen/Qwen3.5-35B-A3B trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6; do
        step "download $m"
        "$VENV/bin/hf" download "$m" --exclude "*.pth" > /dev/null
    done
fi
step "done install_wall_s=$(( $(date +%s) - T0 ))"
