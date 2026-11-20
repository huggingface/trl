#!/bin/bash
# OpenRLHF's ProRL v2 math recipe (examples/scripts/train_prorlv2_math_hybrid_engine.sh at v0.11.2), flags copied
# verbatim; only scale knobs are env-overridable. Used for the GPU smoke (tiny scale) and the recipe reproduction.
#   b200-queue --nodes=1 --gres=gpu:8 --exclusive -t 2:00:00 -J orlhf-prorl launch/openrlhf/prorlv2.sbatch
#SBATCH --job-name=orlhf-prorl
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/logs/%x-%j.out
set -uo pipefail

TAG=${TAG:-prorlv2}
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/openrlhf-recipe-${TAG}-${SLURM_JOB_ID}
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/openrlhf/common.sh

SRC=/fsx/amine_dirhoussi/stable-bench-src/openrlhf
MODEL=${MODEL:-$(ls -d $HF_HUB_CACHE/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/ | head -1)}
DATA=$(ls -d $HF_HUB_CACHE/datasets--OpenRLHF--dapo-math-17k/snapshots/*/ | head -1)
EVAL=$(ls -d $HF_HUB_CACHE/datasets--OpenRLHF--aime-2024/snapshots/*/ | head -1)

# Published: 512 prompts x 16 samples, train batch 1024, 100 episodes. Scaled down keeping the 8 optimizer steps per
# rollout ratio (ROLLOUT_BS * N / TRAIN_BS) and everything else as published. MAX_SAMPLES caps the run length.
ROLLOUT_BS=${ROLLOUT_BS:-128}
N=${N:-16}
TRAIN_BS=${TRAIN_BS:-256}
MAX_NEW=${MAX_NEW:-8192}
MAX_SAMPLES=${MAX_SAMPLES:-4096}
EVAL_STEPS=${EVAL_STEPS:-5}

python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node $NGPU \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node $NGPU \
   --vllm.num_engines $NGPU \
   --vllm.tensor_parallel_size 1 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.75 \
   --algo.kl.init_coef 1e-4 \
   --algo.advantage.gamma 1.0 \
   --algo.kl.use_loss \
   --algo.kl.estimator k2 \
   --algo.advantage.estimator reinforce_baseline \
   --algo.dynamic_filtering_enable \
   --algo.dynamic_filtering_range 0 1 \
   --actor.eps_clip_low_high 0.2 0.27 \
   --actor.model_name_or_path $MODEL \
   --reward.remote_url $SRC/examples/python/math_reward_func.py \
   --ckpt.output_dir $RUN_DIR/final \
   --ckpt.path $RUN_DIR/ckpt \
   --ckpt.save_steps -1 \
   --train.batch_size $TRAIN_BS \
   --rollout.batch_size $ROLLOUT_BS \
   --rollout.n_samples_per_prompt $N \
   --train.dynamic_batch_enable \
   --train.num_episodes 1 \
   --data.max_samples $MAX_SAMPLES \
   --data.max_len $((MAX_NEW + 1024)) \
   --rollout.max_new_tokens $MAX_NEW \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 1e-6 \
   --data.prompt_dataset $DATA \
   --data.input_key prompt \
   --data.label_key label \
   --eval.dataset $EVAL \
   --eval.steps $EVAL_STEPS \
   --eval.temperature 1.0 \
   --eval.n_samples_per_prompt 4 \
   --data.apply_chat_template \
   --actor.gradient_checkpointing_enable \
   --ds.packing_samples \
   --vllm.sync_backend nccl \
   --vllm.enforce_eager \
   --vllm.enable_sleep \
   --ds.enable_sleep \
   --algo.advantage.is_correction_level token \
   --algo.advantage.is_correction_mode mask \
   --algo.advantage.is_correction_threshold 0.5 5.0 \
   --train.max_tokens_per_gpu 32768 \
   --reward.stop_properly_penalty_coef 0.0 \
   --logger.tensorboard_dir $RUN_DIR/tb \
   --logger.logging_steps 1 \
   "$@"
RC=$?
echo "JOB_END $(date +%s.%N) rc=$RC" >> "$RUN_DIR/job_times.txt"
exit $RC
