#!/bin/bash
# From-scratch install of OpenRLHF (Ray + vLLM + DeepSpeed) into /fsx/amine_dirhoussi/envs/bench-openrlhf.
# Follows the upstream Dockerfile (dockerfile/Dockerfile at the pinned tag) with uv instead of pip:
#   vllm==0.29.0, transformers==5.15.0, flash-attn==2.8.3 built against vLLM's torch, then openrlhf[vllm].
# Plus the hybrid-model kernels the benchmark requires (flash-linear-attention + fla-core, causal-conv1d).
#
# flash-attn and causal-conv1d are CUDA extensions with no prebuilt wheel for this torch, and hopper-cpu has neither a
# CUDA toolkit nor a C++ compiler, so the whole script runs in a 1-GPU b200 job (the GPU also runs the import check):
#   b200-queue --gres=gpu:1 --cpus-per-task=16 -t 3:00:00 -J orlhf-install \
#     -o /fsx/amine_dirhoussi/stable-bench-runs/openrlhf/install/%x-%j.out envs/openrlhf/install.sh
# Built wheels are shared in /fsx/amine_dirhoussi/stable-bench-src/wheels/torch<ver>-cu<ver>-py<ver>/ and reused when
# they match (always installed with --no-cache: uv's cache can return a wheel built against another torch).
set -euo pipefail
SRC=/fsx/amine_dirhoussi/stable-bench-src/openrlhf
VENV=${VENV:-/fsx/amine_dirhoussi/envs/bench-openrlhf}
SHA=17fd4a79cc87dfb2f6185ffac634066525bcd917   # tag v0.11.2 (latest release, 2026-09-14)
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/openrlhf
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fsx/amine_dirhoussi/.cache/uv}
export UV_HTTP_TIMEOUT=300 UV_LINK_MODE=copy
PY=$VENV/bin/python
T0=$(date +%s)
step() { echo "== [$(( $(date +%s) - T0 ))s] $*"; }

[ -d "$SRC" ] || git clone https://github.com/OpenRLHF/OpenRLHF.git "$SRC"
git -C "$SRC" fetch -q --tags && git -C "$SRC" checkout -q "$SHA"

step "venv"
rm -rf "$VENV"
uv venv --python 3.12 "$VENV"

# 1. vLLM first so its torch pin wins (Dockerfile: "Keep the training and rollout stacks on one resolver-selected
#    PyTorch/CUDA ABI"). cu130 wheels: B200 is sm_100.
step "vllm + pinned trainer deps"
uv pip install --python "$PY" --torch-backend=cu130 \
    "vllm==0.29.0" "deepspeed==0.19.6" "transformers==5.15.0" \
    ninja packaging wheel setuptools psutil einops

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$CUDA_HOME/bin:$VENV/bin:$PATH MAX_JOBS=${SLURM_CPUS_PER_TASK:-16}
TAG=$("$PY" -c "import torch, sys; print('torch' + '.'.join(torch.__version__.split('.')[:2]) + '-cu' + torch.version.cuda.replace('.', '') + f'-py{sys.version_info.major}{sys.version_info.minor}')")
WHEELS=/fsx/amine_dirhoussi/stable-bench-src/wheels/$TAG
mkdir -p "$WHEELS"
step "torch tag $TAG"

# Build (or reuse) a CUDA-extension wheel from a git tag, for sm_100 only, against this venv's torch.
ext_wheel() {  # name repo tag
    local name=$1 repo=$2 tag=$3 whl d
    whl=$(ls "$WHEELS"/${name//-/_}-${tag#v}-*.whl 2>/dev/null | head -1 || true)
    if [ -z "$whl" ]; then
        step "build $name $tag"
        d=$(mktemp -d /tmp/build-$name-XXXX)
        git clone -q --recursive --depth 1 -b "$tag" "$repo" "$d/src"
        (cd "$d/src" && uv build --wheel --no-build-isolation --python "$PY" -o "$d/out" .)
        cp "$d"/out/*.whl "$WHEELS"/
        whl=$(ls "$WHEELS"/${name//-/_}-${tag#v}-*.whl | head -1)
        rm -rf "$d"
    else
        step "reuse $whl"
    fi
    uv pip install --python "$PY" --no-cache --no-deps --reinstall "$whl"
}

# 2. flash-attn 2.8.3 (pinned in requirements.txt), like the Dockerfile's --no-build-isolation rebuild.
FLASH_ATTN_CUDA_ARCHS=100 NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE \
    ext_wheel flash-attn https://github.com/Dao-AILab/flash-attention.git v2.8.3

# 3. OpenRLHF itself with the vllm extra (vllm==0.29.0). Editable so a patch, if ever needed, is a git diff in $SRC.
step "openrlhf[vllm] (editable, pinned tag)"
uv pip install --python "$PY" --torch-backend=cu130 -e "$SRC[vllm]"

# 4. Hybrid (Qwen3.5 gated DeltaNet) kernels + the shared math reward's verifier.
step "fla (with fla-core) + math-verify"
uv pip install --python "$PY" --torch-backend=cu130 flash-linear-attention "math-verify[antlr4_13_2]>=0.8.0" pyarrow
TORCH_CUDA_ARCH_LIST="10.0" CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    ext_wheel causal-conv1d https://github.com/Dao-AILab/causal-conv1d.git v1.7.0

step "import check"
cd /tmp
"$PY" - <<'PY'
import importlib
import torch, vllm, transformers, deepspeed, ray, flash_attn, fla, causal_conv1d, openrlhf, math_verify
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from causal_conv1d import causal_conv1d_fn
from flash_attn import flash_attn_varlen_func
print("fla chunk_gated_delta_rule ->", chunk_gated_delta_rule.__module__)
for m in (torch, vllm, transformers, deepspeed, ray, flash_attn, fla, causal_conv1d):
    print(f"{m.__name__:16s} {getattr(m, '__version__', '?')}")
print("torch cuda", torch.version.cuda, "triton", importlib.import_module("triton").__version__)
x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
print("causal_conv1d on GPU OK:", tuple(causal_conv1d_fn(x, w, activation="silu").shape))
q = torch.randn(256, 8, 128, device="cuda", dtype=torch.bfloat16)
cu = torch.tensor([0, 100, 256], device="cuda", dtype=torch.int32)
print("flash_attn varlen on GPU OK:", tuple(flash_attn_varlen_func(q, q, q, cu, cu, 156, 156, causal=True).shape))
import openrlhf.cli.train_ppo_ray  # the trainer entry point imports cleanly
PY
uv pip freeze --python "$PY" > "$HERE/freeze.txt"
step "done install_wall_s=$(( $(date +%s) - T0 )) venv_gb=$(du -sb "$VENV" | awk '{printf "%.1f", $1/1e9}')"
