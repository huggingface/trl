#!/bin/bash
#SBATCH --job-name=serve_protein_qwen_vllm_dp_4g
#SBATCH --partition=vector_gpu_priority
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512gb
#SBATCH --output=serve_protein_qwen_vllm_dp_%j_%x.out
#SBATCH --error=serve_protein_qwen_vllm_dp_%j_%x.err

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

# ---------- Per-job caches (local, writable) ----------
SCRATCH_JOB="/home/parsaidp/trl/vllm_eval_dp"
export TMPDIR="$SCRATCH_JOB/tmp"
export TEMP="$SCRATCH_JOB/tmp"
export TMP="$SCRATCH_JOB/tmp"
export TRANSFORMERS_CACHE="$SCRATCH_JOB/transformers"
export WANDB_DIR="$SCRATCH_JOB/wandb"
# export HF_HOME="$SCRATCH_JOB/hf_home"
# export HF_DATASETS_CACHE="$SCRATCH_JOB/hf_datasets"

mkdir -p "$TMPDIR" "$TRANSFORMERS_CACHE" "$WANDB_DIR" 

# Make repo importable
export PYTHONPATH="$ROOT_DIR/trl:$PYTHONPATH"
export PYTHONPATH="$ROOT_DIR/BioReason2:$PYTHONPATH"

# ==============================================================================
# Config (aligned with your existing run_eval_vllm.sh)
# ==============================================================================
TEXT_MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
PROTEIN_MODEL_NAME="esm3-sm-open-v1"
GO_OBO_PATH="${ROOT_DIR}/BioReason2/data/go-basic.obo"
GO_EMBEDDINGS_PATH="/large_storage/goodarzilab/bioreason/go_embeddings"
SERVE_SCRIPT="$ROOT_DIR/trl/trl/scripts/vllm_serve.py"

# vLLM knobs (match run_eval_vllm.sh defaults)
DTYPE="auto"
KV_CACHE_DTYPE="auto"
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.7
TRUST_REMOTE_CODE=true
ENABLE_PREFIX_CACHE=false
ENFORCE_EAGER=true
BATCH_INFER=true
SKIP_ENV_CHECK=true
TOKENIZER_PATH=""

# Dataset config (match eval script and training defaults)
DATASET_CACHE_DIR="/large_storage/goodarzilab/parsaidp/data/"
STRUCTURE_DIR="/large_storage/goodarzilab/bioreason/data/structures/"
MAX_LENGTH_PROTEIN=2000

# Ports and devices for 4 single-GPU servers
HOST="0.0.0.0"
PORTS=(8001 8002 8003 8004)
DEVICES=(0 1 2 3)

set -euo pipefail

# Launch 4 servers, each pinned to a single GPU and its own server log
pids=()
for i in 0 1 2 3; do
  PORT=${PORTS[$i]}
  DEV=${DEVICES[$i]}
  SERVER_LOG="$SCRATCH_JOB/server_${PORT}.log"
  mkdir -p "$(dirname "$SERVER_LOG")"
  echo "Launching server on GPU $DEV port $PORT → logs $SERVER_LOG"

  CUDA_VISIBLE_DEVICES=$DEV \
  python "$SERVE_SCRIPT" \
    --model "/home/adibvafa/BioReason2/checkpoints/esm3-qwen-4B-finetune-mixed-Qwen3-4B-Thinking-2507-split_go_aspects_4b_2507_lr1e-4_32gpus-stage2/last-v1-hf.ckpt" \
    --tokenizer "${TOKENIZER_PATH:-$TEXT_MODEL_NAME}" \
    --host "$HOST" \
    --port "$PORT" \
    --log_level "info" \
    --tensor_parallel_size 1 \
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
    > "$SERVER_LOG" 2>&1 &
  pids+=("$!")
done

echo "Server PIDs: ${pids[*]}"

# Simple health poll for each server
healthcheck() {
  local port=$1
  curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 || \
  curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

echo "Waiting for servers to become healthy…"
deadline=$((SECONDS + 300))
for PORT in "${PORTS[@]}"; do
  until healthcheck "$PORT"; do
    # Ensure process is still alive
    alive=false
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=true; break; fi
    done
    if [ "$alive" = false ]; then
      echo "A server process died during startup. Check logs in $SCRATCH_JOB"; exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timeout waiting for server on port $PORT"; exit 1
    fi
    sleep 2
  done
  echo "Port $PORT healthy."
done

# Round-robin eval client across ports (same flags as eval_cafa_vllm.py)
echo "Starting round-robin eval across: ${PORTS[*]}"
python "$ROOT_DIR/trl/trl/scripts/eval_cafa_vllm_dp.py" \
  --hosts 127.0.0.1 \
  --ports ${PORTS[*]} \
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
  --val_split_ratio 0.9 \
  --seed 23 \
  --max_length_protein "$MAX_LENGTH_PROTEIN" \
  --max_samples 128 \
  --request_batch_size 4 \
  --concurrent_requests 8 \
  --save_results \
  --first_batch_out "$SCRATCH_JOB/batches_0.json" \
  --results_out "$SCRATCH_JOB/cafa_vllm_results.json"

# Keep job alive until all servers exit
wait


