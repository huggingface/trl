#!/bin/bash
# From-scratch install of verl (FSDP + vLLM, the backend combo the verl docs recommend) into
# /fsx/amine_dirhoussi/envs/bench-verl. Runs in a hopper-cpu job:
#   sbatch -p hopper-cpu --qos=low -c 16 --mem=32G -t 3:00:00 -o /fsx/amine_dirhoussi/stable-bench-runs/verl/logs/install-%j.out envs/verl/install.sh
set -euo pipefail
SRC=/fsx/amine_dirhoussi/stable-bench-src/verl
VENV=/fsx/amine_dirhoussi/envs/bench-verl
SHA=1876b06d0a3e4e71e06230be10af14492ca8a75b   # tag v0.9.1 (latest release, 2026-09-20)
T0=$(date +%s)

[ -d "$SRC" ] || git clone https://github.com/volcengine/verl.git "$SRC"
git -C "$SRC" fetch -q --tags && git -C "$SRC" checkout -q "$SHA" && git -C "$SRC" submodule update --init -q recipe
# Patch 1 (the only one): v0.9.1's V1 separate_async with hybrid_engine=False calls logger.info(..., flush=True), which
# raises TypeError at trainer init. Upstream removed the argument in ace775e8 (#7926, 2026-09-22, not in a release yet).
# Dropped 2026-11-10 (user: no library patches): the line below used to apply that fix.
# sed -i 's/^                flush=True,$//' "$SRC/verl/trainer/ppo/v1/trainer_separate_async.py"
git -C "$SRC" diff --stat
rm -rf "$VENV"

# 1. verl's own recommended install: sync the committed universal uv.lock with the vllm + fsdp extras
#    (docs/start/install.rst). The project venv goes outside the repo via UV_PROJECT_ENVIRONMENT.
#    The fsdp extra already pins flash-linear-attention==0.5.2 (+ fla-core 0.5.2) and flash-attn 2.8.3 (verl wheelhouse).
cd "$SRC"
#    flash-attn is skipped here and installed in step 1b: the lock's wheel URL (release tag `flash-attention-v2.8.3`)
#    404s since the verl wheelhouse renamed that release to `flash-attention-v2.8.3-py3.12-torch2.11.0` (same build).
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --all-packages --python 3.12 --extra vllm --extra fsdp --extra math \
  --no-install-package flash-attn
# 1b. the same flash-attn 2.8.3 cu130/torch2.11 wheel from its renamed release (sha256 from the GitHub release).
FA_URL=https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.11.0/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
FA_SHA=578253f946f93be01b31c89f02d6ab8d7d9d52c3d6130e9fef4ade1d8b750b4c
curl -sSfL -o /tmp/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl "$FA_URL"
echo "$FA_SHA  /tmp/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" | sha256sum -c -
uv pip install --python "$VENV/bin/python" --no-deps /tmp/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
T1=$(date +%s)

# 2. causal-conv1d is not in verl's lock and has no prebuilt wheel for torch 2.11, so it must be compiled. hopper-cpu
#    nodes have no C++ compiler or CUDA toolkit, so that step is envs/verl/build_causal_conv1d.sh, run on a GPU node after this.
T2=$(date +%s)

# 3. import check
"$VENV/bin/python" - <<'PY'
import torch, vllm, transformers, verl, fla, flash_attn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "transformers", transformers.__version__, "verl", verl.__version__)
print("fla", fla.__version__, "flash_attn", flash_attn.__version__)
PY
T3=$(date +%s)
echo "TIMING sync_s=$((T1-T0)) import_s=$((T3-T2)) total_s=$((T3-T0))"
du -sh "$VENV"
