#!/bin/bash

# Description: Submits a Slurm job with flexible command-line arguments.
#
# Usage:
# ./submit_job.sh [OPTIONS] <path_to_job_script> <array_size>
#
# OPTIONS:
#   -n <num>      Number of nodes (default: 4)
#   -t <hours>    Wall time limit in hours (default: 2)
#   -d <str>      Job dependency string (e.g., "afterok:123456")
#   -h            Display this help message

# --- nrt-specific envs ---
export NCCL_IB_TIMEOUT=20 # timeout=4.096 µs * 2 ^ NCCL_IB_TIMEOUT; NCCL_IB_TIMEOUT=20 --> timeout ~4.3s
export NCCL_IB_SL=0
export NCCL_IB_TC=41 # Sets the traffic class. For OCI, TC41 maps to their priority queue.
export NCCL_IGNORE_CPU_AFFINITY=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=4 # Scale-dependent, =4 is suggested for small (<2048 GPU) jobs by OCI. Observed =1 is fine at all scales.
export NCCL_CROSS_NIC=0 # 0 for rail-optimized, 1 for fat-tree/cross-rail
export NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_10,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
export RX_QUEUE_LEN=8192
export IB_RX_QUEUE_LEN=8192
export NCCL_DEBUG=WARN # Must be WARN for FACT attribution. Don't allow override.
export NCCL_NET_PLUGIN=none # Must be explicitly set to avoid containers trying to use SHARP breaking on RoCE

# --- 1. Set Default Values ---
NUM_NODES=4
TIME_LIMIT=2
DEPENDENCY=""
PARTITION="batch_block1,batch_short"

# --- 2. Define Usage Function (Updated) ---
usage() {
    echo "Usage: $0 [-n num_nodes] [-t time_limit_hr] [-d dependency] [-p partition] <path_to_job_script> <array_size>"
    echo "  -n: Number of nodes (default: ${NUM_NODES})"
    echo "  -t: Time limit in hours (default: ${TIME_LIMIT})"
    echo "  -d: Job dependency (e.g., 'afterok:123456')"
    echo "  -p: Partition (default: ${PARTITION})"
    echo "  <path_to_job_script>: (Required) Path to the sbatch script to run."
    echo "  <array_size>:         (Required) The upper bound of the job array (e.g., 10 for --array=1-10)."
    echo "  -h: Display this help message"
    exit 1
}

# --- 3. Parse Optional Flag Arguments (-n, -t, -d) ---
# The loop continues as long as the first argument starts with a "-"
while getopts "n:t:d:p:h" opt; do
  case ${opt} in
    n) NUM_NODES=${OPTARG} ;;
    t) TIME_LIMIT=${OPTARG} ;;
    d) DEPENDENCY=${OPTARG} ;;
    p) PARTITION=${OPTARG} ;;
    h) usage ;;
    \?) usage ;; # Handle invalid options
  esac
done
shift "$((OPTIND - 1))" # Remove the parsed options from the argument list

# --- 4. Handle Positional Arguments (JOB_PATH, ARRAY) (Updated) ---
if [[ -z "$1" ]] || [[ -z "$2" ]]; then
    echo "Error: Both <path_to_job_script> and <array_size> are required."
    usage
fi
JOB_PATH=$1
ARRAY=$2

# --- 5. Build and Submit the Job (Original Logic) ---
TMP="${JOB_PATH%.sh}"
JOB_NAME="${TMP#scripts/}"
LEN=${#JOB_NAME}
SHORT_JOB_NAME=""

if [[ $LEN -gt 76 ]]; then
    SHORT_JOB_NAME="${JOB_NAME:0:20}...${JOB_NAME: -56}"
else
    SHORT_JOB_NAME="${JOB_NAME}"
fi

echo ">> Preparing to submit job with:"
echo "   - Nodes: ${NUM_NODES}"
echo "   - Time Limit: ${TIME_LIMIT}h"
echo "   - Array Size: 1-${ARRAY}"
[[ -n "$DEPENDENCY" ]] && echo "   - Dependency: ${DEPENDENCY}"

# SLURM Options Construction
SBATCH_OPTIONS=""
SBATCH_OPTIONS+=" -A nvr_elm_llm"                               # account
SBATCH_OPTIONS+=" -p ${PARTITION}"                  # partition
SBATCH_OPTIONS+=" -t ${TIME_LIMIT}:00:00"                       # wall time limit, hr:min:sec
SBATCH_OPTIONS+=" -N ${NUM_NODES}"                              # number of nodes
SBATCH_OPTIONS+=" -J ${SHORT_JOB_NAME}"     # job name
SBATCH_OPTIONS+=" --array 1-${ARRAY}%1"
SBATCH_OPTIONS+=" --gpus-per-node 8"
SBATCH_OPTIONS+=" --exclusive"
if [[ -n $DEPENDENCY ]]; then
    SBATCH_OPTIONS+=" --dependency=afterany:${DEPENDENCY}"
fi

echo ">> Submitting job with command:"
CMD="sbatch --export=ALL ${SBATCH_OPTIONS} ${JOB_PATH}"
echo "${CMD}"

# Uncomment the line below to actually submit the job
eval ${CMD}