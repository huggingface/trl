# hello world experiment
python benchmark/benchmark.py \
    --command "python examples/scripts/ppo.py --log_with wandb" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template

python benchmark/benchmark.py \
    --command "python examples/scripts/dpo.py --model_name_or_path=gpt2 --per_device_train_batch_size 4 --max_steps 1000 --learning_rate 1e-3 --gradient_accumulation_steps 1 --logging_steps 10 --eval_steps 500 --output_dir="dpo_anthropic_hh" --optim adamw_torch --warmup_steps 150 --report_to wandb --bf16 --logging_first_step --no_remove_unused_columns" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template

python benchmark/benchmark.py \
    --command "python examples/scripts/sft.py --model_name_or_path="facebook/opt-350m" --report_to="wandb" --learning_rate=1.41e-5 --per_device_train_batch_size=64 --gradient_accumulation_steps=16 --output_dir="sft_openassistant-guanaco" --logging_steps=1 --num_train_epochs=3 --max_steps=-1 --push_to_hub --gradient_checkpointing" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template

python benchmark/benchmark.py \
    --command "python examples/scripts/reward_modeling.py --model_name_or_path=facebook/opt-350m --output_dir="reward_modeling_anthropic_hh" --per_device_train_batch_size=64 --num_train_epochs=1 --gradient_accumulation_steps=16 --gradient_checkpointing=True --learning_rate=1.41e-5 --report_to="wandb" --remove_unused_columns=False --optim="adamw_torch" --logging_steps=10 --eval_strategy="steps" --max_length=512" \
    --num-seeds 3 \
    --start-seed 1 \
    --workers 10 \
    --slurm-nodes 1 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 12 \
    --slurm-template-path benchmark/trl.slurm_template
