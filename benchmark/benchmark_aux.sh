## w/ and w/o gradient accumulation
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_step_grad_accu --ppo_config.mini_batch_size 1 --ppo_config.gradient_accumulation_steps 128 --ppo_config.log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template

# compound
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_gpt2xl_grad_accu --ppo_config.model_name gpt2-xl --ppo_config.mini_batch_size 16 --ppo_config.gradient_accumulation_steps 8 --ppo_config.log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template

## w/ different models (gpt2, gpt2-xl, falcon, llama2)
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_gpt2 --ppo_config.log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_falcon_rw_1b --ppo_config.model_name tiiuae/falcon-rw-1b --ppo_config.log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template


## w/ and w/o PEFT
python benchmark/benchmark.py \
    --command "python examples/scripts/sentiment_tuning.py --ppo_config.exp_name sentiment_tuning_peft --use_peft --ppo_config.log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template