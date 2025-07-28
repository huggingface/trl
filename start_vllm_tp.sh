#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  start_vllm_tp.sh — High‑throughput Protein‑LLM server (single engine, TP=4)
# -----------------------------------------------------------------------------
#  • Loads the 1.7 B Qwen‑ESM3 checkpoint ONCE and shards it across 4 GPUs with
#    Tensor Parallel (TP).
#  • Keeps dynamic batching global (better token/​s) and leaves NCCL P2P / IB
#    enabled for fast cross‑GPU all‑reduce.
#  • Supports either text‑only, DNA, or protein multimodal modes via flags.
#  • Creates a fast‑tokenizer folder on first launch if missing.
# -----------------------------------------------------------------------------
#  Usage examples
#  -------------
#  ./start_vllm_tp.sh                             # default protein mode
#  ./start_vllm_tp.sh dna                         # DNA mode
#  ./start_vllm_tp.sh text                        # text‑only
# -----------------------------------------------------------------------------

set -euo pipefail

# ───────── CONFIG ────────────────────────────────────────────────────────────
MODEL_PATH="/large_storage/goodarzilab/parsaidp/last_cafa_1.7B_ESM3"
DNA_MODEL_PATH="esm3_sm_open_v1"      # also used for protein (ESM‑3)
HOST="0.0.0.0"
PORT=8000                              # single engine → single port
GPU_MEMORY_UTILIZATION=0.85           # we own the whole GPU
MAX_MODEL_LEN=5000
VLLM_MAX_BATCH=64                     # export to allow bigger micro‑batch
LOG_LEVEL="info"

# mode: protein | dna | text (default protein)
MODE="${1:-protein}"

# ───────── INFO ──────────────────────────────────────────────────────────────
echo "🚀 Starting single vLLM engine with TP=1 …"
echo "   Model           : $MODEL_PATH"
echo "   Mode            : $MODE"
echo "   Visible GPUs    : $(nvidia-smi --query-gpu=index,name --format=csv,noheader)"
echo "   Host/Port       : $HOST:$PORT"

# ───────── RUN ───────────────────────────────────────────────────────────────

export VLLM_MAX_REQUEST_BATCH_SIZE=$VLLM_MAX_BATCH
# Important: *do not* disable NCCL P2P — we need it for TP

CUDA_VISIBLE_DEVICES=0 \
python -m trl.scripts.vllm_serve \
  --model "$MODEL_PATH" \
  $( [[ "$MODE" == "protein" ]] && echo "--protein_model_name $DNA_MODEL_PATH --use_protein_llm" ) \
  $( [[ "$MODE" == "dna"      ]] && echo "--dna_model_name $DNA_MODEL_PATH --use_dna_llm" ) \
  --tensor_parallel_size 1 \
  --data_parallel_size 1 \
  --host "$HOST" \
  --port $PORT \
  --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
  --max_model_len $MAX_MODEL_LEN \
  --dtype auto \
  --enable_prefix_caching false \
  --kv_cache_dtype auto \
  --trust_remote_code true \
  --batch_inference true \
  --log_level $LOG_LEVEL \
  --skip_env_check
