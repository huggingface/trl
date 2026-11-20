#!/bin/bash
# Env: KEY (state/eval key prefix, default skyrl9b), MAX_B200_JOBS (skip submitting while I already hold that many b200 jobs).
# Queues the 12-sample local held-out W2 eval (the x12 setup of results/tables/w2_heldout_x12.md, stable-bench-runs/queue/x12/
# driver.sh) for each HF export of a SkyRL recommended W2 run as it lands. Idempotent (state/<key>.waiter), requeue-safe.
#   sbatch -p hopper-cpu --qos=low -c 1 --mem=2G -t 7-0 -J skyrl-w2rec-evals -o /fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out \
#       launch/skyrl/w2rec_evals.sh <run id> "20 40 60"
for v in $(compgen -e | grep -E '^(SLURM|SBATCH|SRUN)_'); do unset "$v"; done
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
RUN=$1; STEPS=${2:-${STEPS:-20 40 60}}
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN
ST=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/w2rec/state; mkdir -p $ST
PY=/fsx/amine_dirhoussi/envs/bench-trl-w2/bin/python
log() { echo "$(date -u +%FT%TZ) $*"; }
while true; do
  left=0
  for N in $STEPS; do
    k=${KEY:-skyrl9b}-$N
    [ -s $ST/$k.waiter ] && continue
    left=1
    EXP=$RUN_DIR/global_step_$N/policy
    # complete: config + weights present, nothing written for 3 min (bash: the python heredoc check failed on hopper-cpu, 2026-11-14)
    [ -f "$EXP/config.json" ] && ls "$EXP"/*.safetensors >/dev/null 2>&1 && [ -z "$(find "$EXP" -newermt '-3 minutes' -print -quit)" ] || continue
    # node budget (audit reruns): at most 2 of my hopper-b200 jobs at once, so an eval waits for a free slot
    [ -n "${MAX_B200_JOBS:-}" ] && [ "$(squeue -u $USER -p hopper-b200 -h -t R | wc -l)" -ge "$MAX_B200_JOBS" ] && continue
    out=$ST/$k.submit.log
    ( cd $BENCH && env STAGE=probe SIZE=M NVLLM=4 SPLIT=eval SANDBOX=local HARBOR_SHARDS=4 TERMINUS_MAX_TURNS=1 SAMPLES=12 \
        EVAL_MODEL="$EXP" TAG="x12-$k" PRIO_TAG=PRIO_TRAIN \
        /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:4 -t 4:00:00 -J "w2local-evalx12-$k" launch/trl/w2.sbatch > "$out" 2>&1 ) &
    pid=$!; until grep -q 'Submitted batch job' "$out" || ! kill -0 $pid 2>/dev/null; do sleep 2; done
    grep -oP 'Submitted batch job \K[0-9]+' "$out" > $ST/$k.waiter
    log "queued $k (waiter $(cat $ST/$k.waiter)) $EXP"
  done
  [ $left = 0 ] && { log "all queued"; wait; break; }
  sleep 300
done
