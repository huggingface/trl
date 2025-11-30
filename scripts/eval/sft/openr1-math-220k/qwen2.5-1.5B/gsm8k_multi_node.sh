#!/bin/bash
#SBATCH -A nvr_elm_llm                      #account
#SBATCH -p batch_large,batch,batch_short    #partition
#SBATCH -t 02:00:00                         #wall time limit, hr:min:sec
#SBATCH -N 4                                #number of nodes
#SBATCH -J nvr_elm_llm-ellm:pretrain        #job name
#SBATCH --array=1-10%1
#SBATCH --gpus-per-node 8
#SBATCH --exclusive

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

export OMP_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

PROMPT_TYPE="qwen25-math-cot"
MODEL_TAG="sft/openr1-math-220k/qwen2.5-1.5B/checkpoint-500"
MODEL_NAME_OR_PATH="results/${MODEL_TAG}"
OUTPUT_DIR=results/eval/${MODEL_TAG}

DATA_NAME="gsm8k"
SPLIT="test"
NUM_TEST_SAMPLE=-1

read -r -d '' cmd <<EOF
torchrun --nnodes $SLURM_NNODES --nproc_per_node=$gpu_count --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
trl/evaluation/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --evaluate_max_workers 32
EOF

if [ -n "$SLURM_JOB_ID" ] && [ ! -t 0 ]; then
    # on nv cluster
    echo ${cmd}
    srun bash -c "${cmd}"
else
    # on local machine
    echo ${cmd}
    bash -c "${cmd}"
fi
