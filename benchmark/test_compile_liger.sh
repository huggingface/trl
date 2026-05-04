#!/bin/bash
# Test: SFT with liger + compile vs liger + no compile
# This eliminates the logits overhead (liger skips entropy/accuracy from logits)
source /fsx/amine_dirhoussi/trl/.venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export HF_HOME=/fsx/amine_dirhoussi/.cache
export HF_HUB_OFFLINE=1

CONFIG=/fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml

COMMON_ARGS="--model_name_or_path Qwen/Qwen3-30B-A3B \
  --dataset_name THUDM/LongAlign-10k \
  --max_length 16384 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing true \
  --attn_implementation sdpa \
  --packing --packing_strategy wrapped \
  --max_steps 20 --logging_steps 5 \
  --include_num_input_tokens_seen true \
  --save_strategy no \
  --dtype bfloat16 --tf32 true \
  --report_to none \
  --use_liger_kernel true"

echo "========================================="
echo "Test: $1"
echo "========================================="

if [ "$1" = "liger_compile" ]; then
  accelerate launch --config_file $CONFIG \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    trl/scripts/sft.py $COMMON_ARGS \
    --output_dir /tmp/bench_liger_compile \
    --torch_compile
elif [ "$1" = "liger_eager" ]; then
  accelerate launch --config_file $CONFIG \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    trl/scripts/sft.py $COMMON_ARGS \
    --output_dir /tmp/bench_liger_eager
elif [ "$1" = "noliger_compile" ]; then
  accelerate launch --config_file $CONFIG \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    trl/scripts/sft.py $COMMON_ARGS \
    --use_liger_kernel false \
    --output_dir /tmp/bench_noliger_compile \
    --torch_compile
elif [ "$1" = "noliger_eager" ]; then
  accelerate launch --config_file $CONFIG \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    trl/scripts/sft.py $COMMON_ARGS \
    --use_liger_kernel false \
    --output_dir /tmp/bench_noliger_eager
fi
