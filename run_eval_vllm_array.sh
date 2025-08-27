#!/bin/bash
#SBATCH --job-name=serve_protein_qwen_vllm_replicas
#SBATCH --partition=vector_gpu_priority
#SBATCH --time=12:00:00
#SBATCH --nodes=1                 # 1 node per array task
#SBATCH --gpus=4                  # tensor parallel size = 4 per node
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=512gb
#SBATCH --array=0-2               # <-- 3 replicas (use 0-1 for two replicas)
#SBATCH --output=serve_%A_%a.out
#SBATCH --error=serve_%A_%a.err

set -eo pipefail
USER=adibvafa
ENV_NAME=bio

# ==============================================================================
# Environment Setup (matches your cluster)
# ==============================================================================
export PATH=/home/$USER/miniconda/envs/$ENV_NAME/bin:$PATH
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate $ENV_NAME
ROOT_DIR=/home/$USER
cd "$ROOT_DIR/trl/trl"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset SLURM_TRES_PER_TASK

# ---------- Shared storage ----------
SCRATCH_BASE="/large_storage/goodarzilab/bioreason"
SCRATCH_JOB="/home/$USER/trl/vllm_eval"

# Per-replica index/ports/paths
IDX="${SLURM_ARRAY_TASK_ID}"
PORT=$((8000 + IDX))
REPL_DIR="$SCRATCH_JOB/replica_${IDX}"
mkdir -p "$REPL_DIR"

# Per-replica dirs (avoid collisions)
export TMPDIR="$REPL_DIR/tmp"
export TEMP="$REPL_DIR/tmp"
export TMP="$REPL_DIR/tmp"
export TRANSFORMERS_CACHE="$REPL_DIR/transformers"
export WANDB_DIR="$REPL_DIR/wandb"
mkdir -p "$TMPDIR" "$TRANSFORMERS_CACHE" "$WANDB_DIR"

# Make repo importable (vLLM worker extension path resolution)
export PYTHONPATH="$ROOT_DIR/trl:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/BioReason2:$PYTHONPATH"

# Helpful NCCL defaults for single node
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

# ==============================================================================
# Config (aligned with your training config)
# ==============================================================================
TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
# NOTE: If you get a lookup error, try the underscore variant: esm3_sm_open_v1
PROTEIN_MODEL_NAME="esm3-sm-open-v1"

GO_OBO_PATH="$ROOT_DIR/BioReason2/data/go-basic.obo"
GO_EMBEDDINGS_PATH="/large_storage/goodarzilab/bioreason/go_embeddings"

# Dataset/eval config to match training
DATASET_CACHE_DIR="/large_storage/goodarzilab/bioreason/data/"
STRUCTURE_DIR="/large_storage/goodarzilab/bioreason/data/structures/"
MAX_LENGTH_PROTEIN=2000

# Paths
SERVE_SCRIPT="$ROOT_DIR/trl/trl/scripts/vllm_serve.py"
EVAL_SCRIPT="$ROOT_DIR/trl/trl/scripts/eval_cafa_vllm.py"

# vLLM knobs
DTYPE="auto"
KV_CACHE_DTYPE="auto"
MAX_MODEL_LEN=8000  # 4096
GPU_MEM_UTIL=0.75 #  0.88
TRUST_REMOTE_CODE=true
ENABLE_PREFIX_CACHE=false
ENFORCE_EAGER=true
BATCH_INFER=true        # batched embedding→text integration
SKIP_ENV_CHECK=true     # avoid interactive prompt

# Optional fast tokenizer dir (folder with tokenizer.json). Leave empty to auto-build.
TOKENIZER_PATH=""
REVISION=""

# Eval toggles
RUN_EVAL="${RUN_EVAL:-true}"       # set RUN_EVAL=false to keep only servers
KEEP_SERVER="${KEEP_SERVER:-false}"# set true to keep server running after eval
WARMUP_SECS="${WARMUP_SECS:-75}"   # warm-up time before health checks
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-300}"

# Derive tokenizer argument
if [[ -n "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_ARG="${TOKENIZER_PATH}"
else
  TOKENIZER_ARG="${TEXT_MODEL_NAME}"
fi

# Detect GPUs to set tensor_parallel_size = min(4, visible)
detect_gpus () {
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "$SLURM_GPUS_ON_NODE"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    python - <<'PY'
import os
v=os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
print(len([x for x in v.split(",") if x!=""])) if v else print(0)
PY
  else
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  fi
}
NUM_GPUS=$(detect_gpus || echo 4)
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
TENSOR_PARALLEL_SIZE=$(( NUM_GPUS < 4 ? NUM_GPUS : 4 ))
echo "[Replica $IDX] GPUs visible: $NUM_GPUS -> tensor_parallel_size=$TENSOR_PARALLEL_SIZE"

# Logs
SERVER_LOG="$REPL_DIR/server.log"
EVAL_LOG="$REPL_DIR/eval.log"

# Cleanup on exit
trap 'echo "[Replica $IDX] Cleanup: kill $SERVER_PID"; [[ -n "${SERVER_PID:-}" ]] && kill -TERM "$SERVER_PID" 2>/dev/null || true' EXIT

# ------------------------------------------------------------------------------
# Launch server in the background
# ------------------------------------------------------------------------------
echo "[Replica $IDX] Starting server on port $PORT ..."
set -x
srun --ntasks=1 \
  python "$SERVE_SCRIPT" \
    --model "/home/adibvafa/BioReason2/checkpoints/esm3-qwen-4B-finetune-mixed-Qwen3-4B-Thinking-2507-split_go_aspects_4b_2507_lr1e-4_32gpus-stage2/last-v1-hf.ckpt" \
    --tokenizer "$TOKENIZER_ARG" \
    --host "0.0.0.0" --port "$PORT" --log_level "info" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" --data_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" --kv_cache_dtype "$KV_CACHE_DTYPE" \
    --max_model_len "$MAX_MODEL_LEN" \
    --trust_remote_code "$TRUST_REMOTE_CODE" \
    --enable_prefix_caching "$ENABLE_PREFIX_CACHE" \
    --enforce_eager "$ENFORCE_EAGER" \
    --batch_inference "$BATCH_INFER" \
    --skip_env_check "$SKIP_ENV_CHECK" \
    --use_dna_llm false --use_protein_llm true \
    --protein_model_name "$PROTEIN_MODEL_NAME" \
    --go_obo_path "$GO_OBO_PATH" \
    --precomputed_go_embeddings_path "$GO_EMBEDDINGS_PATH" \
    --go_embedding_dim 2560 --go_hidden_dim 512 \
    --go_num_gat_layers 3 --go_num_heads 8 \
    --go_num_reduced_embeddings 200 \
    ${REVISION:+--revision "$REVISION"} \
    > "$SERVER_LOG" 2>&1 &
set +x
SERVER_PID=$!
echo "[Replica $IDX] Server PID: $SERVER_PID (logs: $SERVER_LOG)"

# ------------------------------------------------------------------------------
# Health check loop
# ------------------------------------------------------------------------------
sleep "$WARMUP_SECS"
echo "[Replica $IDX] Polling health for up to ${HEALTH_TIMEOUT_SECS}s ..."
deadline=$((SECONDS + HEALTH_TIMEOUT_SECS))
healthcheck() {
  curl -fsS "http://127.0.0.1:${PORT}/health"    >/dev/null 2>&1 || \
  curl -fsS "http://127.0.0.1:${PORT}/health/"   >/dev/null 2>&1 || \
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}
until healthcheck; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[Replica $IDX] Server died during startup. Tail of log:"
    tail -n 200 "$SERVER_LOG" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "[Replica $IDX] Server failed health checks. Tail of log:"
    tail -n 200 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 2
done
echo "[Replica $IDX] ✅ Server healthy on 127.0.0.1:${PORT}"

# ------------------------------------------------------------------------------
# Optional: run evaluator locally against this replica
# ------------------------------------------------------------------------------
if [[ "$RUN_EVAL" == "true" ]]; then
  echo "[Replica $IDX] Running evaluator ..."
  set -x
  python "$EVAL_SCRIPT" \
    --host "127.0.0.1" \
    --port "$PORT" \
    --cafa5_dataset "wanglab/cafa5" \
    --cafa5_dataset_name "experiment_data" \
    --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --structure_dir "$STRUCTURE_DIR" \
    --include_go_defs False \
    --interpro_dataset_name "interpro_metadata" \
    --split_go_aspects True \
    --interpro_in_prompt True \
    --ppi_in_prompt True \
    --include_protein_function_summary True \
    --val_split_ratio 0.1 \
    --seed 23 \
    --max_length_protein "$MAX_LENGTH_PROTEIN" \
    --max_samples 128 \
    --request_batch_size 64 \
    --concurrent_requests 4 \
    --temperature 0 \
    --top_p 1 \
    --max_new_tokens 6000 \
    --repetition_penalty 1.0 \
    --save_results \
    --first_batch_out "$REPL_DIR/batches.json" \
    --results_out "$REPL_DIR/results.json" \
    > "$EVAL_LOG" 2>&1
  set +x
  echo "[Replica $IDX] Eval done. Results: $REPL_DIR/results.json"
fi

# ------------------------------------------------------------------------------
# Keep or stop server
# ------------------------------------------------------------------------------
if [[ "$KEEP_SERVER" == "true" ]]; then
  echo "[Replica $IDX] KEEP_SERVER=true → keeping the server alive until job timeout."
  wait "$SERVER_PID"
else
  echo "[Replica $IDX] Stopping server (KEEP_SERVER=false)."
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
fi

echo "[Replica $IDX] ✅ Completed."
