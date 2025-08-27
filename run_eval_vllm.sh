#!/bin/bash
#SBATCH --job-name=serve_protein_qwen_vllm_1n4g
#SBATCH --partition=vector_gpu_priority
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4                 # cap at 4 (script auto-sets TP=min(4, visible))
#SBATCH --ntasks-per-node=1      # one server process
#SBATCH --cpus-per-task=24
#SBATCH --mem=512gb
#SBATCH --output=serve_protein_qwen_vllm_%j_%x.out
#SBATCH --error=serve_protein_qwen_vllm_%j_%x.err

# ==============================================================================
# Environment Setup (matches your cluster)
# ==============================================================================
export PATH=/home/parsaidp/miniconda/envs/esmenv/bin:$PATH
source /home/parsaidp/miniconda/etc/profile.d/conda.sh
conda activate esmenv
ROOT_DIR=/home/parsaidp
cd "$ROOT_DIR/trl/trl"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset SLURM_TRES_PER_TASK

# ---------- Use shared lab storage for all caches ----------
SCRATCH_BASE="/large_storage/goodarzilab/parsaidp"
SCRATCH_JOB="/home/parsaidp/trl/vllm_eval"

# Per-job dirs
export TMPDIR="$SCRATCH_JOB/tmp"
export TEMP="$SCRATCH_JOB/tmp"
export TMP="$SCRATCH_JOB/tmp"
export TRANSFORMERS_CACHE="$SCRATCH_JOB/transformers"
export WANDB_DIR="$SCRATCH_JOB/wandb"


mkdir -p "$TMPDIR" "$TRANSFORMERS_CACHE" \
        "$WANDB_DIR" 

# If you use a venv instead of conda PATH, uncomment:
# source .venv/bin/activate

# Make repo importable (vLLM worker extension path resolution)
export PYTHONPATH="$ROOT_DIR/trl:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/BioReason2:$PYTHONPATH"

# ==============================================================================
# Config (aligned with your training config)
# ==============================================================================
TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
PROTEIN_MODEL_NAME="esm3-sm-open-v1"

GO_OBO_PATH="${ROOT_DIR}/BioReason2/data/go-basic.obo"
GO_EMBEDDINGS_PATH="/large_storage/goodarzilab/bioreason/go_embeddings"

# Dataset/eval config to match training
DATASET_CACHE_DIR="/large_storage/goodarzilab/parsaidp/data/"
STRUCTURE_DIR="/large_storage/goodarzilab/bioreason/data/structures/"
MAX_LENGTH_PROTEIN=2000

# Path to the vLLM serve script (the long script you shared)
SERVE_SCRIPT="$ROOT_DIR/trl/trl/scripts/vllm_serve.py"

# vLLM knobs
DTYPE="auto"
KV_CACHE_DTYPE="auto"
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.8
TRUST_REMOTE_CODE=true
ENABLE_PREFIX_CACHE=false
ENFORCE_EAGER=false
BATCH_INFER=true        # enables batched-embeddings path in ProteinEmbeddingProcessor
SKIP_ENV_CHECK=true     # avoid interactive prompt

# Optional fast tokenizer dir (folder with tokenizer.json). Leave empty to auto-build.
TOKENIZER_PATH=""
REVISION=""

# ==============================================================================
# Derive tensor_parallel_size = min(4, #visible GPUs)
# ==============================================================================
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

NUM_GPUS=$(detect_gpus)
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
TENSOR_PARALLEL_SIZE=$(( NUM_GPUS < 4 ? NUM_GPUS : 4 ))
echo "GPUs visible: $NUM_GPUS -> tensor_parallel_size=$TENSOR_PARALLEL_SIZE"

# Helpful NCCL defaults for single node
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

# ==============================================================================
# Launch single server (Protein mode) — robust wait-then-eval
# ==============================================================================
set -euo pipefail

PORT=8000
HOST="0.0.0.0"
HEALTH_HOST="127.0.0.1"
RUN_EVAL=${RUN_EVAL:-true}

# Eval size control: 0 means use entire validation set
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# Optional: short warmup before polling (seconds)
WARMUP_SECS="${WARMUP_SECS:-75}"            # <- set 60–120 as you like
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-300}"  # total wait after warmup

# If TOKENIZER_PATH is set, prefer it; otherwise use TEXT_MODEL_NAME
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  TOKENIZER_ARG="$TOKENIZER_PATH"
else
  TOKENIZER_ARG="$TEXT_MODEL_NAME"
fi

# Don’t collide with another process on the same port
if command -v ss >/dev/null 2>&1 && ss -ltn | grep -q ":$PORT "; then
  echo "ERROR: Port $PORT already in use. Exiting."
  exit 1
fi

# Clean up server on any exit
trap 'echo "[CLEANUP] Terminating server (PID=${SERVER_PID:-})"; \
      [[ -n "${SERVER_PID:-}" ]] && kill -TERM "$SERVER_PID" 2>/dev/null || true' EXIT

set -x
SERVER_LOG="$SCRATCH_JOB/server.log"
mkdir -p "$(dirname "$SERVER_LOG")"

# --- Start server in the background under the Slurm allocation ---
srun --ntasks=1 \
  python "$SERVE_SCRIPT" \
    --model "/home/adibvafa/BioReason2/checkpoints/esm3-qwen-4B-finetune-mixed-Qwen3-4B-Thinking-2507-split_go_aspects_4b_2507_lr1e-4_32gpus-stage2/last-v1-hf.ckpt" \
    --tokenizer "$TOKENIZER_ARG" \
    --host "$HOST" \
    --port "$PORT" \
    --log_level "info" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --data_parallel_size 1 \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --kv_cache_dtype "$KV_CACHE_DTYPE" \
    --max_model_len "$MAX_MODEL_LEN" \
    --trust_remote_code "$TRUST_REMOTE_CODE" \
    --enable_prefix_caching "$ENABLE_PREFIX_CACHE" \
    --enforce_eager "$ENFORCE_EAGER" \
    --batch_inference "$BATCH_INFER" \
    --skip_env_check "$SKIP_ENV_CHECK" \
    --use_dna_llm false \
    --use_protein_llm true \
    --protein_model_name "$PROTEIN_MODEL_NAME" \
    --go_obo_path "$GO_OBO_PATH" \
    --precomputed_go_embeddings_path "$GO_EMBEDDINGS_PATH" \
    --go_embedding_dim 2560 \
    --go_hidden_dim 512 \
    --go_num_gat_layers 3 \
    --go_num_heads 8 \
    --go_num_reduced_embeddings 200 \
    ${REVISION:+--revision "$REVISION"} \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
set +x
echo "Server PID: $SERVER_PID (logs: $SERVER_LOG)"

# Optional: stream early logs so you see compilation/startup
( sleep 3; tail -n +1 -f "$SERVER_LOG" & ) >/dev/null 2>&1 || true

# --- Warmup then health-poll ---
echo "Warming up for ${WARMUP_SECS}s to let vLLM load kernels/model…"
sleep "$WARMUP_SECS"

healthcheck() {
  # Return 0 when healthy (try a few plausible endpoints)
  curl -fsS "http://$HEALTH_HOST:$PORT/health"   >/dev/null 2>&1 || \
  curl -fsS "http://$HEALTH_HOST:$PORT/health/"  >/dev/null 2>&1 || \
  curl -fsS "http://$HEALTH_HOST:$PORT/v1/models" >/dev/null 2>&1
}

echo "Polling health for up to ${HEALTH_TIMEOUT_SECS}s…"
deadline=$((SECONDS + HEALTH_TIMEOUT_SECS))
until healthcheck; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process died during startup."
    echo "---- last 200 lines of server log ----"
    tail -n 200 "$SERVER_LOG" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Server failed health checks within ${HEALTH_TIMEOUT_SECS}s."
    echo "---- last 200 lines of server log ----"
    tail -n 200 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 2
done
echo "Server is healthy."

# --- Run eval if requested ---
if [ "$RUN_EVAL" = true ]; then
  echo "Running eval client (trl/scripts/eval_cafa_vllm.py)…"
  python "$ROOT_DIR/trl/trl/scripts/eval_cafa_vllm.py" \
    --host "$HEALTH_HOST" \
    --port "$PORT" \
    --cafa5_dataset "wanglab/cafa5" \
    --cafa5_dataset_name "cafa5_reasoning" \
    --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --structure_dir "$STRUCTURE_DIR" \
    --include_go_defs False \
    --interpro_dataset_name interpro_metadata \
    --split_go_aspects True \
    --interpro_in_prompt True \
    --ppi_in_prompt True \
    --include_protein_function_summary True \
    --val_split_ratio 0.1 \
    --seed 23 \
    --max_length_protein "$MAX_LENGTH_PROTEIN" \
    --max_samples 128 \
    --request_batch_size 64 \
    --concurrent_requests 2 \
    --debug True \
    --save_results \
    --first_batch_out "$SCRATCH_JOB/batches_0.json" \
    --results_out "$SCRATCH_JOB/cafa_vllm_results.json"
fi

# Keep the job alive while server runs (until killed or timed out)
wait "$SERVER_PID"
