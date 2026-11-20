#!/bin/bash
# CPU check of the NeMo-RL W1 setup (check_w1.py) inside the NeMo-RL container. hopper-cpu.
#SBATCH -J nemo-rl-w1-check
#SBATCH -p hopper-cpu
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --mem=0 --container-image=$IMG --container-mounts=/fsx:/fsx,/scratch:/scratch \
  --container-workdir=/opt/nemo-rl --export=ALL bash -c "
export PATH=/opt/nemo_rl_venv/bin:\$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
export PYTHONPATH=$B/launch/nemo-rl/w1:$B/workloads/math TMPDIR=/tmp/n\$SLURM_JOB_ID RAY_TMPDIR=/tmp/n\$SLURM_JOB_ID
mkdir -p /tmp/n\$SLURM_JOB_ID
python -c 'import math_verify, importlib.metadata as m; print(\"math_verify\", m.version(\"math-verify\"), \"antlr\", m.version(\"antlr4-python3-runtime\"))'
cd $B/launch/nemo-rl/w1 && python check_w1.py \${CHECK_CONFIG:-$B/configs/nemo-rl/w1-S.yaml} 2>&1 | grep -v Warning | tail -40
"
