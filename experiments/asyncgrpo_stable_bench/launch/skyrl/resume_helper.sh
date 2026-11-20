#!/bin/bash
# Waits for a SkyRL GPU job to end; if it ended by TIMEOUT / NODE_FAIL / PREEMPTED (not COMPLETED, not a watcher kill,
# not a manual cancel), resubmits the same sbatch with RUN_ID=<same run> so SkyRL resumes from its latest checkpoint
# (resume_mode=latest), then waits on the new job. Up to 3 resubmits.
#   sbatch -p hopper-cpu --qos=low -c 1 --mem=2G -t 7-0 launch/skyrl/resume_helper.sh <job id> <run id> <sbatch> <time>
for v in $(compgen -e | grep -E '^(SLURM|SBATCH|SRUN)_'); do unset "$v"; done
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
JOB=$1; RUN=$2; SCRIPT=$3; T=$4; RD=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN
log() { echo "$(date -u +%FT%TZ) $*"; }
for i in 1 2 3; do
  while squeue -h -j $JOB 2>/dev/null | grep -q .; do sleep 120; done
  st=$(sacct -j $JOB -X -n -o State | head -1 | awk '{print $1}')
  log "job $JOB ended: $st"
  case $st in TIMEOUT|NODE_FAIL|PREEMPTED|FAILED) ;; *) log "no resume"; exit 0 ;; esac
  [ -e $RD/WATCH_KILLED ] && { log "watcher kill: $(cat $RD/WATCH_KILLED), no resume"; exit 0; }
  [ "$st" = FAILED ] && ! grep -q "global_step" $RD/ckpts/latest_ckpt_global_step.txt 2>/dev/null && [ ! -s $RD/ckpts/latest_ckpt_global_step.txt ] && { log "failed with no checkpoint, no resume"; exit 0; }
  out=$RD/resume-$i.submit.log
  ( cd $BENCH && RUN_ID=$RUN PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:8 --exclusive -t $T $SCRIPT > $out 2>&1 ) &
  until grep -q 'Submitted batch job' $out; do sleep 5; done
  w=$(grep -oP 'Submitted batch job \K[0-9]+' $out)
  until [ -s /fsx/amine_dirhoussi/stable-bench-runs/queue/state/$w.jid ]; do sleep 30; done
  JOB=$(cat /fsx/amine_dirhoussi/stable-bench-runs/queue/state/$w.jid); log "resume $i: waiter $w, job $JOB (resumes from $(cat $RD/ckpts/latest_ckpt_global_step.txt 2>/dev/null))"
done
