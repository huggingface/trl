#!/bin/bash
#SBATCH -A account
#SBATCH -p partition
#SBATCH -N 5
#SBATCH --gres gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --job-name=trl_nemo_gym
#SBATCH --output=logs/%j/slurm.out
#SBATCH --error=logs/%j/slurm.err

set -euo pipefail

CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:25.06-py3"
MOUNTS="/path/to/mounts:/path/to/mounts"

TRL_DIR="/path/to/trl"
HOME_DIR="/path/to/user"
HF_HOME="/path/to/user/hf_home"
NEMO_GYM_DIR="${TRL_DIR}/examples/scripts/nemo_gym"

MODEL="Qwen/Qwen3-4B-Instruct-2507"

cd "${NEMO_GYM_DIR}"

# 4 training nodes + 1 vLLM node
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
TRAIN_NODE_0="${NODELIST[0]}"
TRAIN_NODE_1="${NODELIST[1]}"
TRAIN_NODE_2="${NODELIST[2]}"
TRAIN_NODE_3="${NODELIST[3]}"
VLLM_NODE="${NODELIST[4]}"

echo "Training Nodes: $TRAIN_NODE_0, $TRAIN_NODE_1, $TRAIN_NODE_2, $TRAIN_NODE_3"
echo "vLLM Node: $VLLM_NODE"

LOG_DIR="logs/${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

# Start vLLM server on the vLLM node
srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-mount-home \
    bash -c "
    set -e
    export HOME=${HOME_DIR}
    export HF_HOME=${HF_HOME}

    LOG_DIR=${NEMO_GYM_DIR}/${LOG_DIR}
    mkdir -p \${LOG_DIR}

    cd ${TRL_DIR}
    source .venv/bin/activate
    python -m trl.scripts.vllm_serve \
        --model ${MODEL} \
        --host 0.0.0.0 \
        --tensor-parallel-size 8 \
        --data-parallel-size 1 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.7 \
        --port 8000 > \${LOG_DIR}/vllm_serve.log 2>&1 &

    wait
" &

echo "Waiting 120s for vLLM server to start."
sleep 120

TRAIN_NODES_LIST="${TRAIN_NODE_0},${TRAIN_NODE_1},${TRAIN_NODE_2},${TRAIN_NODE_3}"

echo "Launching training on ${TRAIN_NODES_LIST}!"

srun --nodes=4 --ntasks=4 --nodelist="${TRAIN_NODES_LIST}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-mount-home \
    bash -c "
    set -e
    export HOME=${HOME_DIR}
    export HF_HOME=${HF_HOME}
    export WANDB_API_KEY=<your_wandb_api_key>

    cd ${TRL_DIR}
    source .venv/bin/activate
    cd examples/scripts/nemo_gym

    accelerate launch \
        --config_file fsdp2.yaml \
        --num_processes 32 \
        --num_machines 4 \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip ${TRAIN_NODE_0} \
        --main_process_port 29500 \
        --rdzv_backend c10d \
        grpo_nemo_gym.py \
        --config config.yaml \
        --vllm_server_host ${VLLM_NODE} \
        --head_server_host ${TRAIN_NODE_0} \
        2>&1 | tee ${NEMO_GYM_DIR}/${LOG_DIR}/training.log
" &

wait
