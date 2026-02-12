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

# Before running this script, set up your Python virtual environments. This only needs to be done once.
# This should be done in from the same container used here.
#
#    cd /path/to/trl
#    uv venv && source .venv/bin/activate && uv sync && uv pip install -e .[vllm] && uv pip install fastapi uvicorn accelerate deepspeed wandb omegaconf
#
#    cd /path/to/Gym
#    uv venv && source .venv/bin/activate && uv sync

CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
MOUNTS="/path/to/mounts:/path/to/mounts"

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

TRAIN_NODE_0="${NODELIST[0]}"
TRAIN_NODE_1="${NODELIST[1]}"
TRAIN_NODE_2="${NODELIST[2]}"
TRAIN_NODE_3="${NODELIST[3]}"
VLLM_NODE="${NODELIST[4]}"

echo "Training Nodes: $TRAIN_NODE_0, $TRAIN_NODE_1, $TRAIN_NODE_2, $TRAIN_NODE_3"
echo "vLLM Node: $VLLM_NODE"
echo "Main process IP: $TRAIN_NODE_0"

LOG_DIR="logs/${SLURM_JOB_ID}"
mkdir -p ${LOG_DIR}

echo "Starting ng_run and vLLM on ${VLLM_NODE}..."
echo "Logs will be saved to: ${LOG_DIR}"

# Start vLLM server and NeMo Gym servers on one node:
srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-mount-home \
    bash -c "
    LOG_DIR=/path/to/logs
    mkdir -p \${LOG_DIR}

    # Start nemo gym servers
    (set -x && \
    export HOME=/path/to/user && \
    cd /path/to/user/Gym && \
    source .venv/bin/activate && \
    ray stop --force && \
    ng_run +config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml] +head_server.host=0.0.0.0 +head_server.port=11000) > \${LOG_DIR}/ng_run.log 2>&1 &

    sleep 10

    # Start trl vllm server
    (set -x && \
    export HOME=/path/to/user && \
    export HF_HOME=/path/to/user/hf_home && \
    cd /path/to/user/trl && \
    source .venv/bin/activate && \
    python -m trl.scripts.vllm_serve \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.7 \
    --port 8000) > \${LOG_DIR}/vllm_serve.log 2>&1 &

    wait
" &

echo "Waiting for nemo gym and vllm to start..."
sleep 120


TRAIN_NODES_LIST="${TRAIN_NODE_0},${TRAIN_NODE_1},${TRAIN_NODE_2},${TRAIN_NODE_3}"

# Launch training on 4 nodes:
srun --nodes=4 --ntasks=4 --nodelist="${TRAIN_NODES_LIST}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-mount-home \
    bash -c "
    set -x && \
    export HOME=/path/to/user && \
    export HF_HOME=/path/to/user/hf_home && \
    cd /path/to/user/trl && \
    source .venv/bin/activate && \
    cd examples/scripts/nemo_gym && \
    export WANDB_API_KEY=<your wandb api key> && \
    accelerate launch \
    --config_file fsdp2.yaml \
    --num_processes 32 \
    --num_machines 4 \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip ${TRAIN_NODE_0} \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    train_multi_environment.py \
    --config config.yaml \
    --vllm_server_host ${VLLM_NODE} \
    --head_server_host ${VLLM_NODE}" &

wait

