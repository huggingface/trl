#!/bin/bash
# W2 (agentic) venv for SkyRL: the W1 stack (SkyRL main d8700468, `uv sync --frozen --extra fsdp`) plus Harbor 0.25.0
# from PyPI with the HF Sandboxes backend, the same Harbor the TRL side runs. A separate venv so bench-skyrl (W1) stays
# untouched: SkyRL's own `harbor` extra pins git 3de07a0e (v0.13.1+25), which has no hf-sandbox (added in v0.21.0).
# Nothing is compiled, so it runs on hopper-cpu:
#   sbatch -p hopper-cpu --qos=low -c 8 --mem=24G --requeue -t 2:00:00 -o /fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out envs/skyrl/install_w2.sh
#SBATCH --job-name=skyrl-install-w2
set -euo pipefail
T0=$(date +%s)
SRC=/fsx/amine_dirhoussi/stable-bench-src/skyrl
VENV=${VENV:-/fsx/amine_dirhoussi/envs/bench-skyrl-w2}
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
SHA=d87004689e3a57e4041991a3c4cf71bed333a44d
export UV_HTTP_TIMEOUT=300
PY=$VENV/bin/python
step() { echo "== [$(( $(date +%s) - T0 ))s] $*"; }

[ "$(git -C "$SRC" rev-parse HEAD)" = "$SHA" ] || { echo "SkyRL checkout is not at $SHA"; exit 1; }

step "venv + uv sync --frozen --extra fsdp (same lock as bench-skyrl)"
rm -rf "$VENV"
uv venv --python 3.12 "$VENV"
cd "$SRC"
# The cache lives on /admin and the venv on /fsx, so hardlinks are impossible: copy.
UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --extra fsdp

# Harbor 0.25.0 + its hf-sandbox extra (huggingface-hub>=1.22; the lock has 1.32.0). The training stack is held at the
# lock's versions with constraints, so only Harbor and its own dependencies move.
step "harbor[hf-sandbox]==0.25.0"
CONSTRAINTS=$(mktemp)
uv pip freeze --python "$PY" | grep -iE '^(torch|vllm|transformers|flash[-_]attn|flashinfer[-_]python|ray|numpy|flash[-_]linear[-_]attention|fla[-_]core|causal[-_]conv1d|tokenizers|omegaconf|antlr4-python3-runtime)==' > "$CONSTRAINTS"
cat "$CONSTRAINTS"
UV_LINK_MODE=copy uv pip install --python "$PY" -c "$CONSTRAINTS" "harbor[hf-sandbox]==0.25.0"
rm -f "$CONSTRAINTS"

# Harbor 0.25.0's HF sandbox upload untars as root without --no-same-owner, so every upload (task environment/, the
# verifier's tests/) fails with "Cannot change ownership". Same patch as the TRL side, written against the 0.25.0 wheel.
step "patch harbor hf_sandbox upload (tar --no-same-owner)"
SITE=$("$PY" -c 'import harbor, os; print(os.path.dirname(harbor.__file__))')
patch -N -p1 -d "$SITE/.." < "$BENCH/workloads/agentic/harbor-0.25.0-hf-sandbox-no-same-owner.patch" || grep -q 'no-same-owner' "$SITE/environments/hf_sandbox.py"

step "import check"
cd "$SRC"
"$PY" - <<'PY'
import inspect
from importlib.metadata import version
import torch, vllm, transformers, ray, skyrl, flash_attn, fla, causal_conv1d, omegaconf, litellm  # noqa: F401
# The Harbor names SkyRL's HarborGenerator imports
from harbor.models.agent.rollout_detail import RolloutDetail  # noqa: F401
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from harbor.environments.hf_sandbox import HFSandboxEnvironment
assert inspect.iscoroutinefunction(Trial.create), "Trial.create is not async"
assert "--no-same-owner" in inspect.getsource(HFSandboxEnvironment.upload_dir), "harbor tar patch missing"
TrialConfig.model_validate({"task": {"path": "/tmp/x"}, "agent": {"name": "terminus-2"}, "environment": {"type": "hf-sandbox"}})
from examples.train_integrations.harbor.harbor_generator import HarborGenerator  # noqa: F401
from examples.train_integrations.harbor.entrypoints.main_harbor_fully_async import HarborFullyAsyncExp  # noqa: F401
for p in ["torch", "vllm", "transformers", "ray", "flash-attn", "flash-linear-attention", "causal-conv1d", "skyrl", "harbor", "litellm", "huggingface-hub", "openai"]:
    print(f"{p}=={version(p)}")
PY
uv pip freeze --python "$PY" > "$BENCH/envs/skyrl/freeze_w2.txt"
step "done install_wall_s=$(( $(date +%s) - T0 )) venv_gb=$(du -sb "$VENV" | awk '{printf "%.1f", $1/1e9}')"
