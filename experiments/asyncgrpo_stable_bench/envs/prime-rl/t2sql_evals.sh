#!/bin/bash
# Greedy evals (miles' eval: T=0, 1 sample per task; TRL's eval.sbatch unchanged) of the prime-rl text-to-SQL run's merged
# adapters, queued as each hf/step_N/merge_done.json lands. Idempotent and requeue-safe (a step with a queue.log is never
# queued twice). Runs in a hopper-cpu job:
#   sbatch -p hopper-cpu --qos=low -c 1 --mem=1G -t 2-0 -J prl-t2sql-evals \
#       -o /fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out envs/prime-rl/t2sql_evals.sh <waiter id> ["<steps>"]
# Results: stable-bench-runs/trl-moe-ep/t2sql-eval/<run id>-checkpoint-N/summary.json (validation, hard, arcwise).
STATE=/fsx/amine_dirhoussi/stable-bench-runs/queue/state/$1.jid; STEPS=${2:-20 40 60 80 100}; EVALS=${EVALS:-validation hard arcwise}
EVAL=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql/data
until [ -s "$STATE" ]; do sleep 120; done
J=$(cat "$STATE")
RUN=prime-rl-t2sql-Qwen3.5-35B-A3B-lora32${RUN_TAG:+-$RUN_TAG}-$J
echo "$(date -u +%FT%TZ) run job $J ($RUN)"
SPECS=""; for e in $EVALS; do SPECS="$SPECS eval-$e=$D/$e.jsonl:1:0"; done
cd $EVAL
declare -A done
n=$(echo $STEPS | wc -w)
while [ ${#done[@]} -lt $n ]; do
  for N in $STEPS; do
    [ -n "${done[$N]:-}" ] && continue
    d=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/$RUN/hf/step_$N
    [ -f "$d/merge_done.json" ] || continue
    OUT=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/t2sql-eval/$RUN-checkpoint-$N
    [ -f $OUT/queue.log ] && { done[$N]=1; continue; }
    mkdir -p $OUT
    echo "$(date -u +%FT%TZ) step $N merged ($d), queueing eval -> $OUT"
    # b200-sbatch counts an unpinned job as a new node; pin to one of my nodes with >= 4 free GPUs when there is one
    W=()
    for node in $(squeue -u $USER -p hopper-b200 -h -t R -o %N | tr ',' '\n' | sort -u); do
      used=$(scontrol show node $node | grep -o 'AllocTRES=[^ ]*' | grep -o 'gres/gpu=[0-9]*' | cut -d= -f2)
      [ "${used:-0}" -le 4 ] && { W=(-w $node); break; }
    done
    PRIO_TAG=PRIO_TRAIN nohup /fsx/amine_dirhoussi/bin/b200-queue "${W[@]}" -J t2sql-eval --gres=gpu:4 -t 4:00:00 \
      --export=ALL,MODEL=$d,OUT=$OUT,NGPU=4,SPECS="$SPECS" $PWD/workloads/text2sql/eval.sbatch > $OUT/queue.log 2>&1 &
    done[$N]=1
  done
  # the job merges its last adapter after training, so give it one more pass after it ends
  [ -n "${ended:-}" ] && { echo "$(date -u +%FT%TZ) run $J ended; queued: ${!done[*]}"; break; }
  squeue -h -j "$J" | grep -q . || ended=1
  sleep 120
done
wait
