#!/bin/bash
# From-scratch install of prime-rl for the stable bench. Runs in a hopper-cpu job:
#   sbatch -p hopper-cpu --qos=low -c 28 --mem=80G -t 3:00:00 -o <log> envs/prime-rl/install.sh
# Follows prime-rl's documented manual setup (README "Manual Setup" + examples/advanced/qwen3-30b-a3b/README.md):
# submodules, then `uv sync --all-extras --all-packages` from the lock. The only deviations are paths:
# the venv lives outside the repo (UV_PROJECT_ENVIRONMENT) and private-SSH submodule URLs go over HTTPS.
set -euo pipefail

SRC=/fsx/amine_dirhoussi/stable-bench-src/prime-rl
SHA=cbd3702237bc7d4e01ed8ff003e46b2a7d56c951 # main on 2026-11-07, 167 commits after v0.10.0
VENV=/fsx/amine_dirhoussi/envs/bench-prime-rl
export UV_PROJECT_ENVIRONMENT=$VENV
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fsx/amine_dirhoussi/.cache/uv}

t0=$(date +%s)
if [ ! -d "$SRC/.git" ]; then
    git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$SRC"
fi
cd "$SRC"
git fetch -q origin
git checkout -q "$SHA"

# .gitmodules uses git@github.com: URLs (needs a GitHub SSH key); route them over HTTPS like scripts/install.sh does.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf" GIT_CONFIG_VALUE_0="git@github.com:"
# prime-envs tracks some files with git-lfs, which the cluster lacks; we need none of them, so skip the smudge.
export GIT_LFS_SKIP_SMUDGE=1
git -c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.required=false \
    submodule update --init --force -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config deps/prime-kernels
git submodule status

# --frozen: install exactly the committed uv.lock (the pyproject's relative `exclude-newer = "7 days"` would otherwise
# re-resolve against a moving cutoff).
uv sync --frozen --all-extras --all-packages
t1=$(date +%s)

"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md, time
t = time.time()
import torch, vllm, transformers, prime_rl, verifiers, fla, flash_attn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # the kernel prime-rl's GatedDeltaNet calls
from fla.modules.conv import causal_conv1d  # the conv prime-rl's GatedDeltaNet calls (Triton, cu_seqlens-aware)
import math_env, reverse_text, aime24, math_verify
print(f"imports ok in {time.time() - t:.1f}s")
for p in ["prime-rl", "torch", "vllm", "transformers", "flash-linear-attention", "fla-core", "causal-conv1d",
          "flash-attn", "flash-attn-4", "verifiers", "renderers", "math-verify", "harbor"]:
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "NOT INSTALLED")
EOF
t2=$(date +%s)
echo "INSTALL_WALL_S sync=$((t1 - t0)) import_ok=$((t2 - t0))"
du -sh "$VENV"
