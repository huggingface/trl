#!/bin/bash
#SBATCH --job-name=trl-gold-buffer-test
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=0-12:00:00

set -euo pipefail

if [[ "$*" == *"--help"* ]]; then
  cat <<'EOF'
Usage: sbatch scripts/slurm/gold.slurm [options]

Required:
  --config PATH             YAML config passed to gold.py

Optional:
  --accelerator NAME|PATH   Accelerate config name (default: zero3) or explicit YAML path
  --args "ARGS"             Extra args appended to gold.py

Examples:
  sbatch scripts/slurm/gold.slurm \
    --config /path/to/config.yaml \
    --accelerator zero3 \
    --args "--bf16 --logging_steps 1"
EOF
  exit 0
fi

module load cuda/12.9 || true
source ~/.bashrc || true

START_TIME=$(date +%s)
echo "START TIME: $(date)"

CONFIG_FILE=""
ACCELERATOR="zero3"
GPUS_PER_NODE=8
OPTIONAL_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --args)
      OPTIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage."
      exit 1
      ;;
  esac
done

source "trl-internal/bin/activate"

if [[ -z "$CONFIG_FILE" ]]; then
  echo "Error: --config is required."
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: config file not found: $CONFIG_FILE"
  exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "Error: accelerate is not available in PATH."
  exit 1
fi

if ! command -v trl >/dev/null 2>&1; then
  echo "Error: trl CLI is not available in PATH."
  exit 1
fi

if [[ -f "$ACCELERATOR" ]]; then
  ACCEL_CONFIG="$ACCELERATOR"
elif [[ -f "trl/accelerate_configs/${ACCELERATOR}.yaml" ]]; then
  ACCEL_CONFIG="trl/accelerate_configs/${ACCELERATOR}.yaml"
elif [[ -f "examples/accelerate_configs/${ACCELERATOR}.yaml" ]]; then
  ACCEL_CONFIG="examples/accelerate_configs/${ACCELERATOR}.yaml"
else
  echo "Error: could not resolve accelerate config from '$ACCELERATOR'."
  exit 1
fi

GRAD_ACC_STEPS=$(grep -E '^\s*gradient_accumulation_steps:' "$CONFIG_FILE" | head -n 1 | awk '{print $2}')
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}

if [[ "$OPTIONAL_ARGS" =~ --gradient_accumulation_steps=([0-9]+) ]]; then
  GRAD_ACC_STEPS="${BASH_REMATCH[1]}"
fi
if [[ "$OPTIONAL_ARGS" =~ --gradient_accumulation_steps[[:space:]]+([0-9]+) ]]; then
  GRAD_ACC_STEPS="${BASH_REMATCH[1]}"
fi

if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "Error: this launcher must run inside a SLURM allocation."
  exit 1
fi

NUM_NODES=${SLURM_NNODES:-1}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=${MASTER_PORT:-6000}
TRAIN_NODES=("${NODELIST[@]}")

NODELIST_CSV=$(IFS=,; echo "${TRAIN_NODES[*]}")

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SCRIPT_PATH="trl/experimental/gold/gold.py"
LAUNCH_CMD="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
  --config_file $ACCEL_CONFIG \
  --gradient_accumulation_steps $GRAD_ACC_STEPS \
  --num_machines $NUM_NODES \
  --num_processes $WORLD_SIZE \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank \$SLURM_PROCID \
  --rdzv_backend=c10d \
  --max_restarts 1 \
  --tee 3 \
  $SCRIPT_PATH --config $CONFIG_FILE $OPTIONAL_ARGS"

SRUN_ARGS=" \
  --wait=60 \
  --kill-on-bad-exit=1 \
  --nodes=$NUM_NODES \
  --ntasks=$NUM_NODES \
  --nodelist=$NODELIST_CSV"

set -x
clear
srun $SRUN_ARGS bash -lc "$LAUNCH_CMD" 2>&1

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"
