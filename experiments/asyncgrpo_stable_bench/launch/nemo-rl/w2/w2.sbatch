#!/bin/bash
# W2 (TermGrade, terminus-2, local Apptainer boxes) on NeMo-RL v0.8.0 (NGC container), Qwen3.5-9B, one exclusive
# 8xB200 node: NeMo-RL's NeMo Gym runner (examples/nemo_gym/run_grpo_nemo_gym.py) with configs/nemo-rl/w2-M.yaml,
# in-job watcher watch.py. Resume: RESUME_DIR=<run dir> reuses that run dir; NeMo-RL restarts from its latest
# checkpoint (every 20 steps).
#   cd .../asyncgrpo_stable_bench && PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:8 --exclusive \
#       -t 30:00:00 launch/nemo-rl/w2/w2.sbatch
#SBATCH -J nemo-rl-w2
#SBATCH --nodes=1
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/nemo-rl/w2
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
export RUN_DIR=${RESUME_DIR:-/fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/nemo-rl-w2-M-local-$SLURM_JOB_ID}
export WORK=/scratch/$USER/nemo-rl-w2-$SLURM_JOB_ID
mkdir -p $RUN_DIR $WORK
rm -f $RUN_DIR/WATCH_KILL
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID on $(hostname)" | tee -a $RUN_DIR/times.txt
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG \
    --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/opt/nemo-rl \
    --export=ALL,RUN_DIR=$RUN_DIR,WORK=$WORK bash $HERE/inner.sh
rc=$?
echo "$(date -u +%FT%TZ) exit $rc" | tee -a $RUN_DIR/times.txt
# Harbor's trial dirs (agent logs, trajectories, verifier output) are node-local: archive them with the run.
tar czf $RUN_DIR/harbor-jobs-$SLURM_JOB_ID.tgz -C $WORK harbor-jobs --exclude='*.cast' 2>/dev/null
pkill -u $USER -f "$WORK" 2>/dev/null; sleep 5; pkill -9 -u $USER -f "$WORK" 2>/dev/null
rm -rf $WORK
exit $rc
