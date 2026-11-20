#!/bin/bash
# GPU smoke / recipe reproduction: SkyRL's published GSM8K recipe (examples/train/gsm8k/run_gsm8k.sh),
# Qwen2.5-1.5B-Instruct, colocated sync GRPO, FSDP + vLLM. Overrides through env:
#   NUM_GPUS (default 2), MAX_STEPS (default 3), TAG (default smoke), MODEL, EXTRA (extra key=value overrides)
#SBATCH --job-name=skyrl-gsm8k
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -uo pipefail

NUM_GPUS=${NUM_GPUS:-2}
MAX_STEPS=${MAX_STEPS:-3}
TAG=${TAG:-smoke}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
DATA_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/gsm8k
RUN_ID=skyrl-gsm8k-${TAG}-${SLURM_JOB_ID}
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN_ID
export TENSORBOARD_DIR=$RUN_DIR/tb
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/skyrl/common.sh

python -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.max_training_steps=$MAX_STEPS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger=tensorboard \
  trainer.enable_ray_gpu_monitor=false \
  data.dataloader.num_workers=0 \
  trainer.project_name=gsm8k \
  trainer.run_name=$RUN_ID \
  trainer.resume_mode=null \
  trainer.log_path="$RUN_DIR/skyrl-logs" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.export_path="$RUN_DIR/exports" \
  ${EXTRA:-} 2>&1 | tee "$RUN_DIR/train.log"
exit ${PIPESTATUS[0]}
