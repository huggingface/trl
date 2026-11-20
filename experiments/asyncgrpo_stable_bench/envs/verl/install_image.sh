#!/bin/bash
# verl's official prebuilt image with the Megatron stack (docs/start/install.rst "Install from docker image";
# docker/Dockerfile.stable.vllm "Target: verlai/verl:vllm024.dev2": torch 2.11 cu130, vLLM 0.24.0, TE v2.16.1,
# Megatron-LM core_v0.18.0, Megatron-Bridge r0.5.0, the same versions as the v0.9.1 lock's megatron extra).
# The image carries no verl (the Dockerfile uninstalls it); verl v0.9.1 comes from the /fsx checkout on PYTHONPATH,
# the docs' "install verl itself without dependencies".
#   sbatch -p hopper-cpu -c 8 --mem=64G -t 3:00:00 -o /fsx/amine_dirhoussi/stable-bench-runs/verl/logs/install-image-%j.out envs/verl/install_image.sh
set -euo pipefail
TAG=${VERL_IMAGE_TAG:-vllm024.dev2}
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/verl-${TAG}.sqsh
T0=$(date +%s)
if [ ! -s "$IMG" ]; then
  rm -f "$IMG.part"
  enroot import -o "$IMG.part" "docker://verlai/verl:${TAG}"
  mv "$IMG.part" "$IMG"
fi
echo "IMPORT_WALL_S=$(( $(date +%s) - T0 ))"; ls -l "$IMG"
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --container-image="$IMG" --container-mounts=/fsx:/fsx bash -c '
export PYTHONPATH=/fsx/amine_dirhoussi/stable-bench-src/verl
python3 - <<PY
import importlib
for m in ["torch","vllm","transformers","verl","megatron.core","megatron.bridge","mbridge","transformer_engine","fla","causal_conv1d","peft","math_verify","flash_attn","transfer_queue"]:
    try:
        mod = importlib.import_module(m); print("OK", m, getattr(mod, "__version__", "?"), getattr(mod, "__file__", ""))
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e)[:200])
PY
pip list 2>/dev/null | grep -i -E "^(torch|vllm|transformers|transformer.engine|megatron|mbridge|flash|fla|causal|peft|math|ray|transferqueue) " 
'
