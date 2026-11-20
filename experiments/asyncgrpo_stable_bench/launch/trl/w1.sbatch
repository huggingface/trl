#!/bin/bash
# W1 (math) on one node: vLLM on the first NVLLM GPUs, the AsyncGRPO trainer (FSDP2) on the rest.
#   timed run:  b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/trl/w1.sbatch
#   smoke:      NVLLM=2 NTRAIN=2 MAX_STEPS=15 TAG=smoke b200-queue --gres=gpu:4 launch/trl/w1.sbatch
# Env knobs: SIZE (S | M: configs/trl/w1-$SIZE.*), ARM (shared | recommended: sources configs/trl/w1-$SIZE-recommended.env), TAG (= ARM), NVLLM (4),
# NTRAIN (4), MAX_STEPS (200), SAVE_STEPS (50), MAX_COMPLETION (16384), ENABLE_THINKING (false), GRAD_ACCUM, TOKEN_BUDGET, MAX_CONCURRENT, ABLATION,
# VLLM_EXTRA.
#SBATCH -J trl-w1
#SBATCH --nodes=1
#SBATCH -t 12:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl/jobs/%x-%j.out
set -uo pipefail
JOB_START=$(date +%s.%N)
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
VENV=/fsx/amine_dirhoussi/envs/bench-trl
SIZE=${SIZE:-S}; export ARM=${ARM:-shared}; TAG=${TAG:-$ARM}; NVLLM=${NVLLM:-4}; NTRAIN=${NTRAIN:-4}
export MAX_COMPLETION=${MAX_COMPLETION:-16384} ENABLE_THINKING=${ENABLE_THINKING-false}  # SHARED_W1 rev. 2
MAX_MODEL_LEN=$((2048 + MAX_COMPLETION))  # max prompt 2048, never truncated (SHARED_W1 rev. 1)
if [ "$ARM" = recommended ]; then set -a; source "$BENCH/configs/trl/w1-$SIZE-recommended.env"; set +a; fi
declare -A MODELS=([S]=Qwen/Qwen3.5-2B [M]=Qwen/Qwen3.5-9B [L]=Qwen/Qwen3.8-27B)
MODEL=${MODELS[$SIZE]}
export RUN_ID=${RESUME_RUN_ID:-trl-w1-$SIZE-$TAG-$SLURM_JOB_ID}  # RESUME_RUN_ID: continue that run from its last checkpoint
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/trl/$RUN_ID
mkdir -p "$RUN_DIR"
echo "$JOB_START" > "$RUN_DIR/job_start"

export PATH=$VENV/bin:$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
export VLLM_URL=http://localhost:$PORT
VLLM_GPUS=$(seq -s, 0 $((NVLLM - 1)))
TRAIN_GPUS=$(seq -s, $NVLLM $((NVLLM + NTRAIN - 1)))

# provenance
cp "$BENCH/configs/trl/w1-$SIZE".* "$0" "$RUN_DIR/" 2>/dev/null
git -C /fsx/amine_dirhoussi/trl-wt-stable-bench rev-parse HEAD > "$RUN_DIR/trl_sha"
git -C /fsx/amine_dirhoussi/trl-wt-stable-bench diff HEAD -- trl > "$RUN_DIR/trl.diff"
uv pip freeze --python $VENV/bin/python > "$RUN_DIR/freeze.txt"
env | grep -E '^(SIZE|ARM|MAX_COMPLETION|ENABLE_THINKING|TAG|NVLLM|NTRAIN|MAX_STEPS|SAVE_STEPS|GRAD_ACCUM|TOKEN_BUDGET|MAX_CONCURRENT|ABLATION|NUM_SAMPLER_LOGPROBS|VLLM_EXTRA)=' > "$RUN_DIR/launch.env"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$RUN_DIR/nodes"

# gpu.csv: 1 Hz per GPU, plus a node column
( echo "node,timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 \
    | sed -u "s/^/$(hostname),/" ) > "$RUN_DIR/gpu.csv" &
SMI_PID=$!

# --generation-config vllm: ignore the model card's sampling defaults (top_p/top_k); the trainer sends its own.
# No images in W1: skip the vision encoder's memory profiling.
CUDA_VISIBLE_DEVICES=$VLLM_GPUS VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --port $PORT --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 \
    --data-parallel-size $NVLLM --api-server-count 8 --seed 42 \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --weight-transfer-config '{"backend":"nccl"}' ${VLLM_EXTRA:-} > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID $SMI_PID 2>/dev/null; pkill -P $$ 2>/dev/null' EXIT

until curl -sf "$VLLM_URL/health" > /dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup"; tail -n 60 "$RUN_DIR/vllm.log"; exit 1; }
    sleep 5
done
echo "vllm_ready_s=$(echo "$(date +%s.%N) - $JOB_START" | bc)" | tee "$RUN_DIR/vllm_ready"

cd "$BENCH"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch --config_file "configs/trl/w1-$SIZE.accelerate.yaml" \
    --num_processes $NTRAIN --main_process_port $((PORT + 7)) "configs/trl/w1-$SIZE.py" 2>&1 | tee "$RUN_DIR/train.log"
rc=${PIPESTATUS[0]}
curl -s "$VLLM_URL/metrics" | grep -E "^vllm:(num_preemptions_total|cache_config_info)" > "$RUN_DIR/vllm_final_metrics.txt"
date +%s.%N > "$RUN_DIR/job_end"
echo "trainer exit code $rc"
exit $rc
