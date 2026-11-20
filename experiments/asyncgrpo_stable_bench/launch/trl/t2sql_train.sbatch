#!/bin/bash
# miles' text-to-SQL recipe with TRL AsyncGRPO on a MoE model, one node: vLLM (native impl, DP=NVLLM + EP, NCCL weight
# transfer) on the first NVLLM GPUs, the trainer (torchrun, DistributedConfig(fsdp_size=NTRAIN, ep_size=EP), LoRA) on the
# next NTRAIN. Episodes run in the trainer's rollout process (SQLite on the node's CPUs, snapshots on /scratch).
# No separate smoke: t2sql_watchdog.py checks the first steps and episodes and merges every checkpoint for the eval.
#   PRIO_TAG=PRIO_TRAIN b200-queue --gres=gpu:8 --exclusive -t 16:00:00 --export=ALL,TRAIN_FILE=...,MAX_STEPS=... $PWD/t2sql_train.sbatch
#SBATCH -J t2sql-trl-moe
#SBATCH --nodes=1
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/jobs/%x-%j.out
set -uo pipefail
JOB_START=$(date +%s.%N)
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
HERE=$BENCH/launch/trl
T2=$BENCH/workloads/text2sql
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
VENV=/fsx/amine_dirhoussi/envs/bench-trl-ep
TRL=/fsx/amine_dirhoussi/trl-wt-asyncgrpo-ep
# RUN_ENV (optional): a file of KEY=VALUE settings decided after the base probe (TRAIN_FILE, GRAD_ACCUM, MAX_STEPS, TAG),
# waited for so the job can be queued before the probe ends
if [ -n "${RUN_ENV:-}" ]; then
    for i in $(seq 360); do [ -f "$RUN_ENV" ] && break; sleep 30; done
    [ -f "$RUN_ENV" ] || { echo "no $RUN_ENV after 3 h"; exit 1; }
    set -a; source "$RUN_ENV"; set +a
fi
export MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B} DTYPE=${DTYPE:-bfloat16} LORA_R=${LORA_R:-32}
NVLLM=${NVLLM:-4}; NTRAIN=${NTRAIN:-4}; export EP=${EP:-$NTRAIN} TAG=${TAG:-t2sql}
export MAX_STEPS=${MAX_STEPS:-100} SAVE_STEPS=${SAVE_STEPS:-20} MAX_STALENESS=${MAX_STALENESS:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-32} TOKEN_BUDGET=${TOKEN_BUDGET:-32768} CONCURRENCY=${CONCURRENCY:-256}
export LR=${LR:-5e-5} WARMUP=${WARMUP:-10} TRAIN_FILE=${TRAIN_FILE:-$D/data/train.jsonl}
export RUN_ID=trl-t2sql-$(basename "$MODEL")-ep$EP-lora$LORA_R-$TAG-$SLURM_JOB_ID
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/$RUN_ID
mkdir -p "$RUN_DIR"
echo "$JOB_START" > "$RUN_DIR/job_start"

export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export PATH=$VENV/bin:$PATH
export PYTHONPATH=$D/river-recipes/text-to-sql/src:$T2:$TRL
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
export VLLM_URL=http://localhost:$PORT
S=/scratch/$USER/t2sql-$SLURM_JOB_ID; mkdir -p -m 700 $S $S/snapshots $S/tmp; chmod 700 $S/snapshots
export VLLM_CACHE_ROOT=$S/vllm-cache

# provenance
cp "$T2"/t2sql_episode.py "$HERE"/t2sql_train.py "$0" "$RUN_DIR/" 2>/dev/null
cp "$TRAIN_FILE" "$RUN_DIR/train_file.jsonl"
git -C $TRL rev-parse HEAD > "$RUN_DIR/trl_sha"; git -C $TRL diff HEAD > "$RUN_DIR/trl.diff"
git -C $D/river-recipes rev-parse HEAD > "$RUN_DIR/river_recipes_sha"
uv pip freeze --python $VENV/bin/python > "$RUN_DIR/freeze.txt"
env | grep -E '^(MODEL|DTYPE|LORA_R|LR|WARMUP|EP|TAG|MAX_STEPS|SAVE_STEPS|MAX_STALENESS|GRAD_ACCUM|TOKEN_BUDGET|CONCURRENCY|TRAIN_FILE)=' > "$RUN_DIR/launch.env"
hostname > "$RUN_DIR/nodes"

( echo "node,timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 \
    | sed -u "s/^/$(hostname),/" ) > "$RUN_DIR/gpu.csv" &
SMI_PID=$!

# One FlashInfer workspace per vLLM venv (envs/trl/NOTES_MOE.md item 31).
FI_WS=/fsx/amine_dirhoussi/flashinfer-ws/$(basename $VENV)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NVLLM - 1))) HF_HUB_OFFLINE=1 VLLM_SERVER_DEV_MODE=1 \
    FLASHINFER_WORKSPACE_BASE=$FI_WS setsid vllm serve "$MODEL" \
    --port $PORT --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --data-parallel-size $NVLLM --enable-expert-parallel --api-server-count $((NVLLM * 2)) --seed 42 \
    --logprobs-mode processed_logprobs --generation-config vllm --return-tokens-as-token-ids \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --weight-transfer-config '{"backend":"nccl"}' ${VLLM_EXTRA:-} > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
stop_vllm() { [ -n "$VLLM_PID" ] || return 0; kill -- -$VLLM_PID 2>/dev/null; sleep 10; kill -9 -- -$VLLM_PID 2>/dev/null; VLLM_PID=""; }
cleanup() { stop_vllm; kill $SMI_PID 2>/dev/null; pkill -P $$ 2>/dev/null; rm -rf "$S"; date +%s.%N > "$RUN_DIR/job_end"; }
trap cleanup EXIT

until curl -sf "$VLLM_URL/health" > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null || grep -q "Engine core initialization failed" "$RUN_DIR/vllm.log"; then
        echo "vLLM died during startup"; tail -n 60 "$RUN_DIR/vllm.log"; exit 1
    fi
    sleep 5
done
echo "vllm_ready_s=$(echo "$(date +%s.%N) - $JOB_START" | bc)" | tee "$RUN_DIR/vllm_ready"

cd "$HERE"
$VENV/bin/python "$HERE/t2sql_watchdog.py" "$RUN_DIR" ${WD_STEPS:-3} ${WD_FIRST_STEP_S:-5400} > "$RUN_DIR/watchdog.log" 2>&1 &
WD_PID=$!
TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$S/snapshots TMPDIR=$S/tmp \
CUDA_VISIBLE_DEVICES=$(seq -s, $NVLLM $((NVLLM + NTRAIN - 1))) $VENV/bin/torchrun --nproc_per_node $NTRAIN \
    --master_port $((PORT + 7)) t2sql_train.py 2>&1 | tee "$RUN_DIR/train.log"
rc=${PIPESTATUS[0]}
echo "trainer exit code $rc"
curl -s "$VLLM_URL/metrics" | grep -E "^vllm:(num_preemptions_total|prefix_cache_(queries|hits)_total|cache_config_info)" > "$RUN_DIR/vllm_final_metrics.txt"
pkill -P $WD_PID 2>/dev/null; kill $WD_PID 2>/dev/null
[ -f "$RUN_DIR/WATCHDOG_FAILED" ] && echo "WATCHDOG FAILED: $(cat "$RUN_DIR/WATCHDOG_FAILED")"
stop_vllm
for ckpt in "$RUN_DIR"/checkpoint-*; do
    [ -d "$ckpt" ] || continue
    out=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/merged/$RUN_ID-$(basename "$ckpt")
    [ -f "$out/merge_check.json" ] && continue
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= $VENV/bin/python "$HERE/merge_lora.py" "$ckpt" "$out" "$MODEL" > "$RUN_DIR/merge-${ckpt##*-}.log" 2>&1
    echo "merge $(basename "$ckpt"): exit $? -> $out"
done
exit $rc
