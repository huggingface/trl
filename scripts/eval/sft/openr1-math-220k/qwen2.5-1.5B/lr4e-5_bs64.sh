#!/bin/bash

if [ -n "$SLURM_JOB_ID" ] && [ ! -t 0 ]; then
    # on nv cluster
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    export LOGLEVEL=INFO

    export PATH="$HOME/workspace/anaconda3/envs/trl/bin:$PATH"
    CODEDIR="$HOME/workspace/code/trl"
    cd $CODEDIR
else
    SLURM_NNODES=1
    head_node_ip=localhost
fi

port=29500

export OMP_NUM_THREADS=32
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_DP_MASTER_IP=$head_node_ip
export VLLM_DP_MASTER_PORT=$port

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

PROMPT_TYPE="qwen25-math-cot"
MODEL_TAG=sft/openr1-math-220k/qwen2.5-1.5B/lr4e-5_bs64
MODEL_NAME_OR_PATH="results/train/${MODEL_TAG}"
JOB_TYPE=eval
OUTPUT_DIR=results/${JOB_TYPE}/${MODEL_TAG}

SPLIT="test"
NUM_TEST_SAMPLE=-1

CKPT_STEP_START=0
CKPT_STEP_INTERVAL=270
CKPT_STEP_END=1350

read -r -d '' cmd_prefix <<EOF
torchrun --nnodes $SLURM_NNODES --nproc_per_node=$gpu_count --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:$port \
trl/evaluation/math_eval.py \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --use_vllm \
    --save_outputs \
    --resume \
    --evaluate_max_workers 32 \
    --wandb_mode online \
    --wandb_group ${MODEL_TAG} \
    --wandb_run_name ${JOB_TYPE}/${MODEL_TAG} \
    --wandb_job_type ${JOB_TYPE}
EOF


cmd_gsm8k="${cmd_prefix} \
    --data_names gsm8k \
    --max_tokens_per_call 8192
"

cmd_math500="${cmd_prefix} \
    --data_names math500 \
    --max_tokens_per_call 8192
"

# load run_single and run_loop functions
setup_cmd="source scripts/eval/setup.sh"

cmds=(
    "run_loop \"${cmd_gsm8k}\" \"${MODEL_NAME_OR_PATH}\" \"${OUTPUT_DIR}\" ${CKPT_STEP_START} ${CKPT_STEP_INTERVAL} ${CKPT_STEP_END}"
    "run_loop \"${cmd_math500}\" \"${MODEL_NAME_OR_PATH}\" \"${OUTPUT_DIR}\" ${CKPT_STEP_START} ${CKPT_STEP_INTERVAL} ${CKPT_STEP_END}"
)


if [ -n "$SLURM_JOB_ID" ] && [ ! -t 0 ]; then
    # on nv cluster
    # loop over cmds and run each sequentially
    for cmd in "${cmds[@]}"; do
        echo $cmd
        srun bash -c "${setup_cmd}; ${cmd}"
    done
else
    # on local machine
    # loop over cmds and run each sequentially
    for cmd in "${cmds[@]}"; do
        echo $cmd
        bash -c "${setup_cmd}; ${cmd}"
    done
fi