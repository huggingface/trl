#!/bin/bash
#SBATCH --job-name=verl-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/%x-%j.out
# GPU smoke: GSM8K GRPO on Qwen2.5-0.5B-Instruct (verl's smallest GSM8K example) through the V1 separate_async
# trainer we use for W1: 2 standalone vLLM GPUs + 2 FSDP2 trainer GPUs, 6 steps.
# Run with: b200-queue launch/verl/smoke.sbatch
set -uo pipefail
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/verl/smoke-$SLURM_JOB_ID
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/verl/common.sh

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
DATA=/fsx/amine_dirhoussi/stable-bench-runs/verl/data/gsm8k

python -m verl.trainer.main_ppo \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.num_warmup_batches=1 \
  trainer.v1.separate_async.parameter_sync_step=2 \
  transfer_queue.enable=True \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=True \
  data.train_files=$DATA/train.parquet \
  data.val_files=$DATA/test.parquet \
  data.train_batch_size=32 \
  data.gen_batch_size=1 \
  data.return_raw_chat=True \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  trainer.logger='["console","file"]' \
  trainer.project_name=verl-smoke \
  trainer.experiment_name=smoke-$SLURM_JOB_ID \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir=$RUN_DIR/ckpt \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=2 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=6 \
  "$@"
RC=$?
echo "EXIT $RC"
exit $RC
