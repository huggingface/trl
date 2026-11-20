#!/bin/bash
# Greedy evals (miles' eval: T=0, 1 sample per task; the same eval.sbatch as TRL's run) of a verl text-to-SQL run's
# merged checkpoints, queued as each merge_check.json lands. Idempotent (a requeued helper skips evals already queued).
# Runs in a hopper-cpu job:
#   sbatch -p hopper-cpu --qos=low -c 1 --mem=1G -t 2-0 -J t2sql-verl-evals -o /fsx/amine_dirhoussi/stable-bench-runs/verl/logs/%x-%j.out \
#       launch/verl/t2sql_evals.sh <waiter id> <tag> "<steps>"
# Results: stable-bench-runs/trl-moe-ep/t2sql-eval/<run id>-checkpoint-N/summary.json
cd /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
STATE=/fsx/amine_dirhoussi/stable-bench-runs/queue/state/$1.jid; TAG=$2; STEPS=$3; EVALS=${4:-validation hard arcwise}
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql/data
until [ -s "$STATE" ]; do sleep 120; done
J=$(cat "$STATE")
RUN=verl-t2sql-Qwen3.5-35B-A3B-lora32-$TAG-$J
echo "$(date -u +%FT%TZ) run job $J ($RUN)"
PREFIX=/fsx/amine_dirhoussi/stable-bench-runs/verl/merged/$RUN-checkpoint
SPECS=""; for e in $EVALS; do SPECS="$SPECS eval-$e=$D/$e.jsonl:1:0"; done
declare -A done
n=$(echo $STEPS | wc -w)
while [ ${#done[@]} -lt $n ]; do
  for N in $STEPS; do
    [ -n "${done[$N]:-}" ] && continue
    d=$PREFIX-$N
    [ -f "$d/merge_check.json" ] || continue
    OUT=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/t2sql-eval/$RUN-checkpoint-$N
    [ -f $OUT/queue.log ] && { done[$N]=1; continue; }
    sleep 60; mkdir -p $OUT
    echo "$(date -u +%FT%TZ) step $N ready ($d), queueing eval -> $OUT"
    PRIO_TAG=PRIO_TRAIN nohup /fsx/amine_dirhoussi/bin/b200-queue -J t2sql-eval --gres=gpu:4 -t 4:00:00 \
      --export=ALL,MODEL=$d,OUT=$OUT,NGPU=4,SPECS="$SPECS" $PWD/workloads/text2sql/eval.sbatch > $OUT/queue.log 2>&1 &
    done[$N]=1
  done
  # the job merges its last checkpoint after training, so give it one more pass after it ends
  [ -n "${ended:-}" ] && { echo "$(date -u +%FT%TZ) run $J ended; queued: ${!done[*]}"; break; }
  squeue -h -j "$J" | grep -q . || ended=1
  sleep 120
done
wait
