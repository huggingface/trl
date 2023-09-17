
# hello world experiment
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template

# # compound
# python benchmark/benchmark.py \
#     --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_gpt2xl_grad_accu --ppo_config.model_name gpt2-xl --ppo_config.mini_batch_size 16 --ppo_config.gradient_accumulation_steps 8 --ppo_config.log_with wandb" \
#     --num-seeds 3 \
#     --start-seed 1 \
#     --workers 10 \
#     --slurm-nodes 1 \
#     --slurm-gpus-per-task 1 \
#     --slurm-ntasks 1 \
#     --slurm-total-cpus 12 \
#     --slurm-template-path benchmark/trl.slurm_template
