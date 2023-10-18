# compound experiments: gpt2xl + grad_accu
python benchmark/benchmark.py \
    --command "python examples/scripts/ppo.py --ppo_config.exp_name ppo_gpt2xl_grad_accu --ppo_config.model_name gpt2-xl --ppo_config.mini_batch_size 16 --ppo_config.gradient_accumulation_steps 8 --ppo_config.log_with wandb" \
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
    --command "accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml examples/scripts/ppo.py --ppo_config.exp_name ppo_Cerebras-GPT-6.7B_grad_accu_deepspeed_stage2  --ppo_config.batch_size 32  --ppo_config.mini_batch_size 32 --ppo_config.log_with wandb --ppo_config.model_name cerebras/Cerebras-GPT-6.7B --ppo_config.reward_model sentiment-analysis:cerebras/Cerebras-GPT-6.7B" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 90 \
    --slurm-template-path benchmark/trl.slurm_template
