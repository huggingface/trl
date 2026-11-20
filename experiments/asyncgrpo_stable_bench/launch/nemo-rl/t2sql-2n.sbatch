#!/bin/bash
# NeMo-RL v0.8.0 on text-to-SQL, two exclusive 8xB200 nodes, with NeMo-RL's own Slurm launcher (ray.sub from the
# v0.8.0 tree, run unchanged: Ray head + worker in the NGC container on each node, then COMMAND on the head).
# Config t2sql/t2sql-2n.yaml (their 2n8g Automodel EP16 recipe + our workload); driver + watcher t2sql/inner.sh.
#   cd .../asyncgrpo_stable_bench && PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --nodes=2 --gres=gpu:8 \
#       --exclusive -t 16:00:00 launch/nemo-rl/t2sql-2n.sbatch
#SBATCH -J nemo-rl-t2sql-2n
#SBATCH --nodes=2
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/nemo-rl
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
T2=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/text2sql
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/t2sql/nemo-rl-t2sql-2n-Qwen3.5-35B-A3B-$SLURM_JOB_ID
export WORK=/scratch/$USER/nemo-rl-t2sql-$SLURM_JOB_ID
mkdir -p $RUN_DIR
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID on $SLURM_JOB_NODELIST" | tee $RUN_DIR/times.txt
# Everything below is exported into the containers on both nodes by ray.sub's srun calls, and from the driver into
# every Ray actor by NeMo-RL's init_ray (runtime_env env_vars = the driver's environment).
export CONTAINER=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh MOUNTS=/fsx:/fsx,/scratch:/scratch
export PATH=/opt/nemo_rl_venv/bin:$PATH   # the container venv first (our login PATH leaks in otherwise)
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export MULTINODE=1 CONFIG=$HERE/t2sql/t2sql-2n.yaml BASE_LOG_DIR=$RUN_DIR
export SETUP_COMMAND="mkdir -p -m 700 $WORK/snapshots"
export COMMAND="bash $HERE/t2sql/inner.sh"
export CPUS_PER_WORKER=92   # ray.sub auto-detects CPUTot=96, but Slurm only gives 92 per node here (CPUEfctv), so its srun step never starts
bash /fsx/amine_dirhoussi/stable-bench-src/nemo-rl/ray.sub
rc=$?
echo "$(date -u +%FT%TZ) exit $rc" | tee -a $RUN_DIR/times.txt
srun --ntasks-per-node=1 rm -rf $WORK
exit $rc
