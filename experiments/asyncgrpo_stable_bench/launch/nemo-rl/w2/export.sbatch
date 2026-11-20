#!/bin/bash
# HF exports of a W2 run's Megatron checkpoints (1 GPU: Megatron-Bridge needs libcuda): NeMo-RL's own examples/converters/convert_megatron_to_hf.py
# in the container, then t2sql/complete_ckpt.py adds the base tensors the export lacks (vision tower / MTP) and the
# tokenizer files, so vLLM loads it like the other libraries' checkpoints. Output: <run dir>/hf/checkpoint-<N>.
#   b200-queue --gres=gpu:1 -t 4:00:00 launch/nemo-rl/w2/export.sbatch <run dir> "20 40 60"
#SBATCH -J nemo-w2-export
#SBATCH -t 4:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
RUN_DIR=$1; STEPS=${2:-20 40 60}
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
T2=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/nemo-rl/t2sql
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --container-image=$IMG --container-mounts=/fsx:/fsx \
  --container-workdir=/opt/nemo-rl bash -c '
export PATH=/opt/nemo_rl_venv/bin:$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
for N in '"$STEPS"'; do
  C='"$RUN_DIR"'/ckpt/step_$N; RAW='"$RUN_DIR"'/hf/raw-$N; OUT='"$RUN_DIR"'/hf/checkpoint-$N
  [ -f $OUT/complete.json ] && continue
  W=$(ls -d $C/policy/weights/iter_* 2>/dev/null | tail -1); [ -n "$W" ] || W=$C/policy/weights
  echo "$(date -u +%FT%TZ) step $N from $W"
  [ -f $RAW/config.json ] || /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python examples/converters/convert_megatron_to_hf.py --config $C/config.yaml \
      --megatron-ckpt-path $W --hf-ckpt-path $RAW 2>&1 | tail -15
  [ -f $RAW/config.json ] || { echo "export of step $N failed"; continue; }
  ALLOW_DTYPE=1 python '"$T2"'/complete_ckpt.py $RAW $OUT Qwen/Qwen3.5-9B 2>&1 | tail -5
done
'
