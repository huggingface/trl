# ruff: noqa
"""Test: actual SFTTrainer with REAL packed data + compile.
Matches the benchmark launch.sh exactly.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen3-30B-A3B \
    --dataset_name THUDM/LongAlign-10k \
    --max_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --packing --packing_strategy wrapped \
    --max_steps 20 --logging_steps 5 \
    --include_num_input_tokens_seen true \
    --save_strategy no \
    --dtype bfloat16 \
    --torch_compile --tf32 true \
    --report_to none \
    --output_dir /tmp/sft_realdata_compile'
"""
# This is just documentation — the actual command is in the docstring above.
# Run it directly via srun.
