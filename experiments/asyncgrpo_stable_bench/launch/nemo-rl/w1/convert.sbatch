#!/bin/bash
# Megatron checkpoint -> complete HF dir for the shared W1 eval, with NeMo-RL's own converter
# (examples/converters/convert_megatron_to_hf.py, run in the container's Megatron worker venv), then
# t2sql/complete_ckpt.py adds any base tensors the export lacks (vision tower, MTP) and the tokenizer/config files.
#   PRIO_TAG=PRIO_TRAIN b200-queue --gres=gpu:1 -t 2:00:00 launch/nemo-rl/w1/convert.sbatch <run dir> "<steps>"
# Writes <run dir>/checkpoint-<step> (the layout workloads/math/eval.sbatch reads).
#SBATCH -J nemo-rl-w1-convert
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -t 4:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
RUN_DIR=$1; STEPS=${2:-50 100 150 200}
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --mem=0 --container-image=$IMG --container-mounts=/fsx:/fsx,/scratch:/scratch \
  --container-workdir=/opt/nemo-rl --export=ALL bash -c "
set -x
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 TMPDIR=/tmp/n\$SLURM_JOB_ID
mkdir -p \$TMPDIR
MPY=\$(ls -d /opt/ray_venvs/*models.policy*MegatronPolicyWorker/bin/python 2>/dev/null | head -1); echo MPY=\$MPY
for N in $STEPS; do
  C=$RUN_DIR/ckpt/step_\$N; OUT=$RUN_DIR/hf/step_\$N
  [ -f $RUN_DIR/checkpoint-\$N/complete.json ] && continue
  IT=\$(ls -d \$C/policy/weights/iter_* | tail -1)
  [ -f \$OUT/config.json ] || \$MPY examples/converters/convert_megatron_to_hf.py --config \$C/config.yaml --megatron-ckpt-path \$IT --hf-ckpt-path \$OUT --no-strict || exit 1
  ALLOW_DTYPE=1 /opt/nemo_rl_venv/bin/python $B/launch/nemo-rl/t2sql/complete_ckpt.py \$OUT $RUN_DIR/checkpoint-\$N Qwen/Qwen3.5-2B || exit 1
done
"
