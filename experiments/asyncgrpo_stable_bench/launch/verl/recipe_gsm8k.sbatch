#!/bin/bash
#SBATCH --job-name=verl-recipe
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/%x-%j.out
# Recipe reproduction: verl's published Qwen2-7B-Instruct GRPO (FSDP2) GSM8K baseline, 89.8 in docs/algo/baseline.md,
# log https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log.
# Every algorithm/batch/parallelism arg below is copied from the command at the top of that log. Changes: paths, loggers,
# total_training_steps=30 (published val plateaus at 0.89-0.90 from step 20), save_freq=-1, val_before_train=True, and
# transfer_queue.enable=True (as in verl's V1 e2e tests). launch/verl/common.sh also raises TransferQueue's ACK timeout (NOTES.md item 20).
# Run with: b200-queue launch/verl/recipe_gsm8k.sbatch
set -uo pipefail
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/verl/recipe-gsm8k-qwen2-7b-$SLURM_JOB_ID
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/verl/common.sh
DATA_PATH=/fsx/amine_dirhoussi/stable-bench-runs/verl/data

python -m verl.trainer.main_ppo \
    transfer_queue.enable=True \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/gsm8k/train.parquet \
    data.val_files=$DATA_PATH/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='verl_grpo_example_gsm8k_record' \
    trainer.experiment_name=qwen2_7b_repro_$SLURM_JOB_ID \
    trainer.default_local_dir=$RUN_DIR/ckpt \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.total_training_steps=30 \
    trainer.total_epochs=15 "$@" 2>&1 | perl -MTime::HiRes=time -ne '$|=1; printf "%.3f %s", time, $_' | tee -a "$RUN_DIR/stdout.log"
RC=${PIPESTATUS[0]}
echo "EXIT $RC"
exit $RC
