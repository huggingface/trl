#!/bin/bash
# NOT PART OF THE BENCHMARK: user rule is verl as shipped, no dependency fixes of ours. Kept only as a note of what it would take (ledger verl-024).
# verl v0.9.1's megatron extra has no flash-linear-attention, but megatron-core's GatedDeltaNet (Qwen3.5) needs it
# (megatron/core/ssm/gated_delta_net.py:44-54, raises when HAVE_FLA is False). Add the version the same lock pins
# for the fsdp extra (flash-linear-attention==0.5.2 + fla-core==0.5.2), deps-free so nothing else moves.
#   sbatch -p hopper-cpu --qos=low -c 4 --mem=16G -t 1:00:00 -o /fsx/amine_dirhoussi/stable-bench-runs/verl/logs/install-mcore-fla-%j.out envs/verl/install_mcore_fla.sh
set -euo pipefail
VENV=/fsx/amine_dirhoussi/envs/bench-verl-mcore
uv pip install --python "$VENV/bin/python" --no-deps flash-linear-attention==0.5.2 fla-core==0.5.2
"$VENV/bin/python" -c "import fla, math_verify; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla', fla.__version__, 'math_verify ok')"
