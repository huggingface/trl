#!/bin/bash
# hopper-cpu: waits for a W2 35B job to end; if global_step_60 was not exported, resubmits w2-35b.sbatch with the same RUN_ID
# (SkyRL resume_mode=latest picks up the last 5-step checkpoint). One resubmission.
#   sbatch -p hopper-cpu --qos=low -c 1 --mem=1G -t 7-0 -J skyrl-w2-35b-resume -o .../%x-%j.out launch/skyrl/w2_35b_resume.sh <job> <run id>
for v in $(compgen -e | grep -E '^(SLURM|SBATCH|SRUN)_'); do unset "$v"; done
J=$1; RUN=$2; BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
while squeue -h -j $J | grep -q .; do sleep 120; done
[ -f /fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN/global_step_60/policy/config.json ] && { echo "$(date -u +%FT%TZ) step 60 exported, done"; exit 0; }
grep -q "KILL" /fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN/watch.log && { echo "$(date -u +%FT%TZ) watcher killed $J, not resuming"; exit 0; }
echo "$(date -u +%FT%TZ) job $J ended ($(sacct -j $J -X -n -o State)), resuming $RUN"
cd $BENCH && RUN_ID=$RUN PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:8 --exclusive -t 16:00:00 launch/skyrl/w2-35b.sbatch
