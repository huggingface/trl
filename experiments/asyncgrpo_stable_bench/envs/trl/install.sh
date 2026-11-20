#!/bin/bash
# Clean install of the TRL (AsyncGRPOTrainer) benchmark venv. Needs a CUDA toolkit for the causal-conv1d build, which
# hopper-cpu nodes don't have, so it runs on hopper-dev (one GPU, also used for the import check):
#   sbatch -p hopper-dev --gres=gpu:1 -c 11 -t 2:00:00 -o $RUNS/install/%x-%j.out envs/trl/install.sh
# Without the causal-conv1d step everything runs on hopper-cpu:
#   sbatch -p hopper-cpu --qos=low -c 8 --mem=24G --requeue -t 2:00:00 -o $RUNS/install/%x-%j.out envs/trl/install.sh
# Every step is timed; the total is install_wall_s in setup.json.
set -euo pipefail
T0=$(date +%s)
VENV=${VENV:-/fsx/amine_dirhoussi/envs/bench-trl}
TRL=${TRL:-/fsx/amine_dirhoussi/trl-wt-stable-bench}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fsx/amine_dirhoussi/.cache/uv}
export UV_HTTP_TIMEOUT=300
PY=$VENV/bin/python
step() { echo "== [$(( $(date +%s) - T0 ))s] $*"; }

# A clean install by default. hopper-cpu is spot: on a requeue (SLURM_RESTART_COUNT > 0) keep the half-built venv,
# since every step below is idempotent. KEEP_VENV=1 does the same by hand.
step "venv"
if { [ "${SLURM_RESTART_COUNT:-0}" = 0 ] && [ "${KEEP_VENV:-0}" = 0 ]; } || [ ! -x "$PY" ]; then
    rm -rf "$VENV"
    uv venv --python 3.12 "$VENV"
fi

# One resolve for everything so vLLM's torch pin wins. cu130 wheels: B200 is sm_100.
# vllm<0.34 is TRL's own upper bound (pyproject `vllm` extra).
step "vllm + trl + deps"
uv pip install --python "$PY" --torch-backend=cu130 \
    "vllm>=0.32,<0.34" \
    -e "$TRL[kernels,peft,math_verify]" \
    "transformers>=5.15" \
    "flash-linear-attention" \
    "math-verify[antlr4_13_2]>=0.8.0" \
    trackio datasets accelerate ninja packaging wheel setuptools

# causal-conv1d ships only an sdist whose setup.py downloads a prebuilt wheel matched to torch/CUDA, and builds
# from source when none matches. Wheels are shared between the benchmark venvs in WHEELS, keyed on torch/CUDA/python:
# reuse one if it's there, otherwise build it (sm_90 + sm_100) against this venv's torch and drop it there.
# Never let uv build it implicitly: with --no-build-isolation uv caches the built wheel by source alone, so a wheel
# built against another torch can come back from the cache (`undefined symbol: _ZN3c104cuda19CUDAErrorLogCaptureC1Ev`).
# vLLM's FlashInfer backend JIT-compiles its kernels at every cold start (~5 min of the first smoke's 15 min vLLM
# startup) unless the JIT cache matching flashinfer-python is installed (flashinfer-cubin has no 0.7.x release).
step "flashinfer precompiled kernels"
FI=$(uv pip show --python "$PY" flashinfer-python | sed -n 's/^Version: //p')
uv pip install --python "$PY" --no-deps "flashinfer-jit-cache==$FI" \
    --extra-index-url https://flashinfer.ai/whl/cu130 --index-strategy unsafe-best-match

step "causal-conv1d"
TAG=$("$PY" -c 'import sys, torch; print("torch%s-cu%s-py%d%d" % (".".join(torch.__version__.split("+")[0].split(".")[:2]), torch.version.cuda.replace(".", ""), *sys.version_info[:2]))')
WHEELS=/fsx/amine_dirhoussi/stable-bench-src/wheels/$TAG
WHL=$(ls "$WHEELS"/causal_conv1d-1.7.0-*.whl 2>/dev/null | head -1 || true)
if [ -z "$WHL" ]; then
    CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
    export CUDA_HOME PATH=$CUDA_HOME/bin:$VENV/bin:$PATH
    export TORCH_CUDA_ARCH_LIST="9.0;10.0" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${SLURM_CPUS_PER_TASK:-8}
    SRC=$(mktemp -d /fsx/amine_dirhoussi/stable-bench-src/causal-conv1d-src.XXXX)
    curl -sL https://files.pythonhosted.org/packages/source/c/causal_conv1d/causal_conv1d-1.7.0.tar.gz | tar xz -C "$SRC"
    mkdir -p "$WHEELS"
    uv build --wheel --no-build-isolation --python "$PY" --out-dir "$WHEELS" "$SRC"/causal_conv1d-1.7.0
    rm -rf "$SRC"
    WHL=$(ls "$WHEELS"/causal_conv1d-1.7.0-*.whl | head -1)
fi
echo "causal-conv1d wheel: $WHL"
uv pip install --python "$PY" --no-cache --reinstall-package causal-conv1d "$WHL"

step "import check"
cd /tmp
"$PY" - <<'PY'
import importlib
import torch, transformers, vllm, trl, fla, causal_conv1d, kernels, math_verify, trackio
from transformers.utils.import_utils import is_flash_linear_attention_available
from transformers.integrations.hub_kernels import resolve_internal_import
fn = resolve_internal_import(importlib.import_module("fla"), "ops.gated_delta_rule.chunk_gated_delta_rule")
print("fla chunk_gated_delta_rule ->", f"{fn.__module__}.{fn.__name__}" if fn else None)
assert fn is not None and fn.__module__.startswith("fla."), "fla does not import: Qwen3.5 would use the torch fallback"
print("causal_conv1d_fn ->", causal_conv1d.causal_conv1d_fn.__module__)
import causal_conv1d_cuda  # the compiled extension: fails here if it was built against another torch
print("causal_conv1d_cuda ->", causal_conv1d_cuda.__file__)
for m in (torch, transformers, vllm, trl, fla, causal_conv1d, kernels, trackio):
    print(f"{m.__name__:16s} {getattr(m, '__version__', '?')}")
print("torch cuda", torch.version.cuda, "triton", importlib.import_module("triton").__version__)
PY
uv pip freeze --python "$PY" > "$TRL/experiments/asyncgrpo_stable_bench/envs/trl/freeze.txt"
step "done install_wall_s=$(( $(date +%s) - T0 )) venv_gb=$(du -sb "$VENV" | awk '{printf "%.1f", $1/1e9}')"
