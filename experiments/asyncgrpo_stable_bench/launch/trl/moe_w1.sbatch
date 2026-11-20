#!/bin/bash
# AsyncGRPO W1 math on a MoE model with expert parallelism + FSDP (sharded at load time by transformers).
#   1 node : NVLLM GPUs serve vLLM, the next NTRAIN GPUs train.     b200-queue --gres=gpu:8 --exclusive moe_w1.sbatch
#   2 nodes: node 0 serves vLLM on 8 GPUs, node 1 trains on 8.      NODES=2: b200-queue --nodes=2 --gres=gpu:8 --exclusive moe_w1.sbatch
# Env: MODEL (Qwen/Qwen3.5-35B-A3B), DTYPE (bfloat16 | float32), LORA_R (0 = full FT), LR, EP (= NTRAIN), NVLLM, NTRAIN, TAG, MAX_STEPS, MAX_STALENESS, MAX_COMPLETION, GRAD_ACCUM,
# TOKEN_BUDGET, SAVE_STEPS, VLLM_EXTRA, VLLM_EP (1: --enable-expert-parallel on the server), VLLM_IMPL (vllm | transformers: --model-impl; vLLM 0.33 cannot serve Qwen3.5 hybrids with transformers, envs/trl/NOTES_MOE.md item 8).
#SBATCH -J trl-moe-w1
#SBATCH -t 6:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/jobs/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/trl
VENV=/fsx/amine_dirhoussi/envs/bench-trl-ep
TRL=/fsx/amine_dirhoussi/trl-wt-asyncgrpo-ep
export MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
NNODES=${SLURM_JOB_NUM_NODES:-1}
if [ "$NNODES" = 2 ]; then NVLLM=${NVLLM:-8}; NTRAIN=${NTRAIN:-8}; else NVLLM=${NVLLM:-4}; NTRAIN=${NTRAIN:-4}; fi
export EP=${EP:-$NTRAIN} TAG=${TAG:-smoke}
export MAX_COMPLETION=${MAX_COMPLETION:-16384} ENABLE_THINKING=${ENABLE_THINKING-false}
MAX_MODEL_LEN=$((2048 + MAX_COMPLETION))
export RUN_ID=trl-moe-w1-$(basename "$MODEL")-ep$EP-${DTYPE:-bfloat16}-lora${LORA_R:-0}-$TAG-$SLURM_JOB_ID
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/$RUN_ID
mkdir -p "$RUN_DIR"
export PATH=$VENV/bin:$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENGINE_READY_TIMEOUT_S=1800
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
VLLM_HOST=${NODELIST[0]}
export VLLM_URL=http://$VLLM_HOST:$PORT

cp "$HERE/moe_w1.py" "$HERE/moe_w1.sbatch" "$RUN_DIR/"
git -C $TRL rev-parse HEAD > "$RUN_DIR/trl_sha"; git -C $TRL diff HEAD > "$RUN_DIR/trl.diff"
env | grep -E '^(MODEL|DTYPE|LORA_R|LR|EP|NVLLM|NTRAIN|TAG|MAX_STEPS|MAX_STALENESS|VLLM_IMPL|MAX_COMPLETION|GRAD_ACCUM|TOKEN_BUDGET|SAVE_STEPS|VLLM_EXTRA|VLLM_EP)=' > "$RUN_DIR/launch.env"
printf "%s\n" "${NODELIST[@]}" > "$RUN_DIR/nodes"

( echo "node,timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 | sed -u "s/^/$(hostname),/" ) > "$RUN_DIR/gpu-vllm.csv" &
SMI_PID=$!

EPFLAG=""; [ "${VLLM_EP:-1}" = 1 ] && EPFLAG="--enable-expert-parallel"
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NVLLM - 1))) HF_HUB_OFFLINE=1 VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --port $PORT --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.85 \
    --data-parallel-size $NVLLM $EPFLAG --api-server-count 8 --seed 42 \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --model-impl ${VLLM_IMPL:-vllm} --weight-transfer-config '{"backend":"nccl"}' ${VLLM_EXTRA:-} > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID $SMI_PID 2>/dev/null; pkill -P $$ 2>/dev/null' EXIT
until curl -sf "$VLLM_URL/health" > /dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup"; tail -n 60 "$RUN_DIR/vllm.log"; exit 1; }
    sleep 5
done
echo "vllm ready after $SECONDS s"

TRAIN_CMD="cd $HERE && torchrun --nproc_per_node $NTRAIN --master_port $((PORT + 7)) moe_w1.py"
if [ "$NNODES" = 2 ]; then
    srun --nodes=1 --ntasks=1 -w "${NODELIST[1]}" --overlap --export=ALL bash -c \
        "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 | sed -u \"s/^/\$(hostname),/\" > $RUN_DIR/gpu-train.csv & $TRAIN_CMD" 2>&1 | tee "$RUN_DIR/train.log"
else
    CUDA_VISIBLE_DEVICES=$(seq -s, $NVLLM $((NVLLM + NTRAIN - 1))) bash -c "$TRAIN_CMD" 2>&1 | tee "$RUN_DIR/train.log"
fi
rc=${PIPESTATUS[0]}
echo "trainer exit code $rc"
# RUN_TESTS=1 (default): reuse the node for the 2-GPU expert-parallel tests once training is done.
if [ "${RUN_TESTS:-1}" = 1 ]; then
    kill $VLLM_PID 2>/dev/null; sleep 20
    ( cd $TRL && CUDA_VISIBLE_DEVICES=0,1 PATH=/fsx/amine_dirhoussi/envs/bench-trl-ep-tests/bin:$PATH \
        HF_HUB_CACHE=/fsx/amine_dirhoussi/hf-hub python -m pytest -p no:cacheprovider -q \
        tests/distributed/test_distributed.py -k async_grpo_expert_parallel ) > "$RUN_DIR/ep-tests.log" 2>&1
    echo "ep tests exit code $? ($(tail -n 1 "$RUN_DIR/ep-tests.log"))"
fi
exit $rc
