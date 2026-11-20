#!/bin/bash
# hopper-cpu: sbatch -p hopper-cpu --qos=low -c 16 --mem=32G -t 1:00:00 envs/skyrl/w2rec_check.sbatch
#SBATCH --job-name=skyrl-w2rec-check
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
VENV=/fsx/amine_dirhoussi/envs/bench-skyrl-w2local
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh
export LD_LIBRARY_PATH=$AP_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /fsx/amine_dirhoussi/stable-bench-src/skyrl; export PYTHONPATH=/fsx/amine_dirhoussi/stable-bench-src/skyrl
T=/scratch/$USER/w2rec-check-$SLURM_JOB_ID; mkdir -p $T
$VENV/bin/python $BENCH/envs/skyrl/w2rec_check.py $BENCH/configs/skyrl/w2-M-recommended.overrides \
  /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train/tasks $T \
  ${TASKS:-task_00156_ce278a task_00193_a89fef task_00228_ceac5a $(ls /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train/tasks | grep -xFf $BENCH/workloads/agentic/nop_pass_tasks.txt | head -1)}
rc=$?; rm -rf $T; exit $rc
