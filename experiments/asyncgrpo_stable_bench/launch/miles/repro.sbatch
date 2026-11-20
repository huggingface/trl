#!/bin/bash -l
#SBATCH --job-name=miles-repro
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
# Recipe reproduction. Target: the only published small-model dapo-math curve in miles,
# examples/infra_features/true_on_policy/README.md ("Qwen3-4B-Base on dapo-math-17k, max_new_tokens = 2048,
# eval aime-2024 with 8192, global batch 64 x 16"; raw_reward.png / aime.png, "baseline" curve).
# Flags = scripts/run_qwen3_4b.py `train --model-name Qwen3-4B-Base` on its default Megatron backend (colocated,
# TP=2 CP=4 on 8 GPUs, rm-type math, KL loss on with coef 0, wd 0.1, betas 0.9/0.98, lr 1e-6), with the README's
# batch (64 prompts x 16, global batch 1024), 2048 response tokens and 8192-token eval. STEPS optimizer steps.
set -uo pipefail
STEPS=${STEPS:-40}
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.1.3.sqsh
RUN=/fsx/amine_dirhoussi/stable-bench-runs/miles/repro-qwen3-4b-base-$SLURM_JOB_ID
mkdir -p $RUN
echo "JOB_START $(date +%s)" | tee $RUN/times.txt
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 \
    | awk -v n=$(hostname) 'NR==1{print $0", node"; fflush(); next}{print $0", "n; fflush()}' > $RUN/gpu.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

# b200-queue submits from a -c 1 --mem=2G waiter with --export=ALL: its SLURM_MEM_PER_NODE / SLURM_CPUS_PER_TASK leak in
# and would cap this srun step at 2 GiB and 1 CPU. Drop them and give the step the whole allocation.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG --container-mounts=/fsx:/fsx --container-workdir=/root/miles \
    --export=ALL,RUN=$RUN,STEPS=$STEPS,PORT_BASE=$((11000 + (SLURM_JOB_ID % 90) * 10)),SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    bash -c '
set -uo pipefail
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles/common.sh
convert_ckpt qwen3-4B $MODELS/Qwen3-4B-Base $CKPT/Qwen3-4B-Base_torch_dist 8 || exit 1
start_ray 8
echo "TRAIN_START $(date +%s)" >> $RUN/times.txt
python3 train.py $(model_args qwen3-4B) \
  --hf-checkpoint $MODELS/Qwen3-4B-Base --ref-load $CKPT/Qwen3-4B-Base_torch_dist \
  --save-debug-event-data $RUN/events \
  --prompt-data $DATA/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt --label-key label --apply-chat-template \
  --rollout-shuffle --rm-type math --num-rollout $STEPS \
  --rollout-batch-size 64 --n-samples-per-prompt 16 --rollout-max-response-len 2048 --rollout-temperature 1 \
  --global-batch-size 1024 --balance-data \
  --eval-interval 20 --eval-prompt-data aime $DATA/aime-2024/aime-2024.jsonl --n-samples-per-eval-prompt 16 \
  --eval-max-response-len 8192 --eval-top-p 1 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 \
  --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl \
  --use-dynamic-batch-size --max-tokens-per-gpu 9216 \
  --rollout-num-gpus-per-engine 1 --sglang-chunked-prefill-size 4096 --sglang-mem-fraction-static 0.7 \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 4 \
  --cp-comm-type a2a --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash --train-memory-margin-bytes 3221225472 \
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate \
  --sglang-context-length 36000 \
  > $RUN/train.log 2>&1
rc=$?
echo "TRAIN_END $(date +%s) EXIT $rc" >> $RUN/times.txt
cp -r /tmp/ray/session_latest/logs $RUN/raylogs 2>/dev/null
stop_ray
exit $rc
'
rc=$?
echo "JOB_END $(date +%s) EXIT $rc" | tee -a $RUN/times.txt
exit $rc
