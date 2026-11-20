#!/bin/bash
# CPU check of the NeMo-RL text-to-SQL glue inside the NeMo-RL container (check_env.py), plus a look at the Automodel
# Qwen3.5-MoE code and package versions. hopper-cpu.
#SBATCH -J nemo-rl-check
#SBATCH -p hopper-cpu
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/nemo-rl/t2sql
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
export WORK=/scratch/$USER/nemo-check-$SLURM_JOB_ID RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/check-$SLURM_JOB_ID
mkdir -p $WORK $RUN_DIR
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --mem=0 --container-image=$IMG --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/opt/nemo-rl \
  --export=ALL,WORK=$WORK,RUN_DIR=$RUN_DIR bash -c '
set -x
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
T2=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/text2sql
export PATH=/opt/nemo_rl_venv/bin:$PATH
ls /opt/ray_venvs; python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
for v in /opt/ray_venvs/*; do echo "== $v"; ls $v/lib/python*/site-packages | grep -i -E "^(vllm|nemo_automodel|transformer_engine|flash_attn|fla|causal_conv1d|torch)-" ; done
A=$(ls -d /opt/ray_venvs/*/lib/python*/site-packages/nemo_automodel/components/models/qwen3_5_moe 2>/dev/null | head -1); echo AUTOMODEL_QWEN35=$A; ls $A
grep -n "nn.Linear\|self\.[a-z_]*proj\|shared_expert" $A/*.py 2>/dev/null | head -40
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
export PYTHONPATH=$D/river-recipes/text-to-sql/src:$T2:'$HERE':'$HERE'/trl_stub
mkdir -p -m 700 $WORK/snapshots /tmp/n$SLURM_JOB_ID; export TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$WORK/snapshots TMPDIR=/tmp/n$SLURM_JOB_ID RAY_TMPDIR=/tmp/n$SLURM_JOB_ID
export T2SQL_TRAIN=$D/data/train.jsonl T2SQL_EPISODES_FILE=$RUN_DIR/episodes.jsonl
cd '$HERE' && python check_env.py ${CHECK_CONFIG:-t2sql.yaml} 2>&1 | tail -60
'
rm -rf $WORK
