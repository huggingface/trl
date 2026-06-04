#!/usr/bin/env bash
# Teacher-server (live) E2E smoke test: a `trl vllm-serve` server + an SDFT/SDPO trainer in server mode with
# use_teacher_server=true. BACKEND in {serial,fsdp2,zero2,zero3}; TRAINER in {sdft,sdpo}; MODE distillation_mode.
set -uo pipefail

ROOT=/fsx/kashif/trl
LOG_DIR="$ROOT/.slurm/pr5862"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PORT=${PORT:-8765}
BACKEND=${BACKEND:-serial}
TRAINER=${TRAINER:-sdft}
MODE=${MODE:-sampled_token}

module load cuda/12.9
export LD_LIBRARY_PATH=/fsx/kashif/miniconda3/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/fsx/kashif/huggingface TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export WANDB_DISABLED=true HF_HUB_ENABLE_HF_TRANSFER=0 TRL_EXPERIMENTAL_SILENCE=1
export NCCL_NET_PLUGIN=none NCCL_NET=Socket NCCL_DEBUG=WARN FLASHINFER_DISABLE_VERSION_CHECK=1

cd "$ROOT"
NGPU=$(nvidia-smi -L | wc -l)
echo "visible GPUs: $NGPU ; backend=$BACKEND trainer=$TRAINER mode=$MODE"

# --- launch server on the last GPU ---
SERVER_GPU=$((NGPU - 1))
SERVER_LOG="$LOG_DIR/tss-server-$SLURM_JOB_ID.log"
CUDA_VISIBLE_DEVICES=$SERVER_GPU trl vllm-serve --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --gpu_memory_utilization 0.4 --enforce_eager true --dtype bfloat16 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID on GPU $SERVER_GPU (log: $SERVER_LOG)"

cleanup() { kill $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

# --- wait for /health/ ---
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/health/" >/dev/null 2>&1; then echo "server healthy after $((i*5))s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED — tail:"; tail -30 "$SERVER_LOG"; exit 1; fi
  sleep 5
done
curl -sf "http://localhost:$PORT/health/" >/dev/null 2>&1 || { echo "server never became healthy"; tail -40 "$SERVER_LOG"; exit 1; }

# --- trainer on the remaining GPU(s) ---
if [ "$TRAINER" = sdft ]; then
  DATA=.slurm/pr5862/datasets/sdft_tiny
  EXTRA="--distillation_alpha 1.0"
else
  DATA=.slurm/pr5862/datasets/sdpo_tiny
  EXTRA="--distillation_weight 1.0 --distillation_alpha 1.0 --success_reward_threshold 0.0 --feedback_from_solution final_answer --include_environment_feedback true --accuracy_eval_num_examples 0"
fi
[ "$MODE" = topk_logits ] && EXTRA="$EXTRA --distillation_topk 20 --distillation_alpha 0.5"
[ "${PEFT:-0}" = 1 ] && EXTRA="$EXTRA --use_peft --lora_r 8 --lora_alpha 16 --lora_target_modules q_proj k_proj v_proj o_proj"

COMMON="--model_name_or_path $MODEL --dataset_path $DATA --max_steps 2 --max_completion_length 8 --num_generations 2 \
  --distillation_mode $MODE --distillation_is_clip None \
  --use_vllm true --vllm_mode server --vllm_server_base_url http://localhost:$PORT \
  --use_teacher_server true --teacher_model_kind live --logging_steps 1 \
  --save_strategy no --eval_strategy no --eval_num_prompts 0 --report_to none $EXTRA"

if [ "$BACKEND" = serial ]; then
  CUDA_VISIBLE_DEVICES=0 python trl/experimental/$TRAINER/$TRAINER.py $COMMON \
    --per_device_train_batch_size 2 --output_dir $LOG_DIR/tss-$TRAINER-$BACKEND-$MODE
else
  case "$BACKEND" in
    fsdp2) CFG=tests/distributed/data/accelerate_configs/fsdp2.yaml ;;
    zero2) CFG=tests/distributed/data/accelerate_configs/zero2.yaml ;;
    zero3) CFG=tests/distributed/data/accelerate_configs/zero3.yaml ;;
  esac
  TRAIN_GPUS=$(seq -s, 0 $((NGPU - 2)))
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python -m accelerate.commands.launch --main_process_port 0 --config_file "$CFG" \
    trl/experimental/$TRAINER/$TRAINER.py $COMMON \
    --per_device_train_batch_size 1 --output_dir $LOG_DIR/tss-$TRAINER-$BACKEND-$MODE
fi
RC=$?
echo "TRAINER EXIT=$RC"
exit $RC
