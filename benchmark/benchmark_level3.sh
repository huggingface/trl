## w/ and w/o gradient accumulation
python benchmark/benchmark.py \
    --command "python examples/scripts/ppo.py --exp_name ppo_step_grad_accu --mini_batch_size 1 --gradient_accumulation_steps 128 --log_with wandb" \
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
    --command "python examples/scripts/ppo.py --exp_name ppo_gpt2 --log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
python benchmark/benchmark.py \
    --command "python examples/scripts/ppo.py --exp_name ppo_falcon_rw_1b --model_name tiiuae/falcon-rw-1b --log_with wandb" \
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
    --command "python examples/scripts/ppo.py --exp_name ppo_peft --use_peft --log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template