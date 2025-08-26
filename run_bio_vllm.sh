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
SCRATCH_JOB="$SCRATCH_BASE/vllm_${SLURM_JOB_ID:-manual}"

# Per-job dirs
export TMPDIR="$SCRATCH_JOB/tmp"
export TEMP="$SCRATCH_JOB/tmp"
export TMP="$SCRATCH_JOB/tmp"
export TRANSFORMERS_CACHE="$SCRATCH_JOB/transformers"
export WANDB_DIR="$SCRATCH_JOB/wandb"
export TRITON_CACHE_DIR="$SCRATCH_JOB/triton_cache"
export TORCH_EXTENSIONS_DIR="$SCRATCH_JOB/torch_extensions"

mkdir -p "$TMPDIR" "$TRANSFORMERS_CACHE" \
        "$WANDB_DIR" "$TRITON_CACHE_DIR" \
         "$TORCH_EXTENSIONS_DIR"

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

GO_OBO_PATH="${ROOT_DIR}/BioReason2/bioreason2/dataset/go-basic.obo"
GO_EMBEDDINGS_PATH="/large_storage/goodarzilab/bioreason/go_embeddings"

# Path to the vLLM serve script (the long script you shared)
SERVE_SCRIPT="$ROOT_DIR/trl/trl/scripts/vllm_serve.py"

# vLLM knobs
DTYPE="auto"
KV_CACHE_DTYPE="auto"
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.88
TRUST_REMOTE_CODE=true
ENABLE_PREFIX_CACHE=false
ENFORCE_EAGER=true
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
# Launch single server (Protein mode)
# ==============================================================================
PORT=8000
HOST="0.0.0.0"

set -x
srun --ntasks=1 \
python "$SERVE_SCRIPT" \
  --model "/home/adibvafa/BioReason2/checkpoints/esm3-qwen-4B-finetune-mixed-Qwen3-4B-Thinking-2507-split_go_aspects_4b_2507_lr1e-4_32gpus-stage2/last-v1-hf.ckpt" \
  --tokenizer "$TEXT_MODEL_NAME" \
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
  ${TOKENIZER_PATH:+--tokenizer "$TOKENIZER_PATH"} \
  ${REVISION:+--revision "$REVISION"}
