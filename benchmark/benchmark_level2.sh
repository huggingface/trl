# compound experiments: gpt2xl + grad_accu
python benchmark/benchmark.py \
    --command "python examples/scripts/ppo.py --exp_name ppo_gpt2xl_grad_accu --model_name gpt2-xl --mini_batch_size 16 --gradient_accumulation_steps 8 --log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template

# compound experiments: Cerebras-GPT-6.7B + deepspeed zero2 + grad_accu
python benchmark/benchmark.py \
    --command "accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml examples/scripts/ppo.py --exp_name ppo_Cerebras-GPT-6.7B_grad_accu_deepspeed_stage2  --batch_size 32  --mini_batch_size 32 --log_with wandb --model_name cerebras/Cerebras-GPT-6.7B --reward_model sentiment-analysis:cerebras/Cerebras-GPT-6.7B" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 90 \
    --slurm-template-path benchmark/trl.slurm_template
