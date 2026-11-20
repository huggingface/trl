#!/bin/bash
# Greedy evals of a NeMo-RL text-to-SQL run's checkpoints with the shared eval (workloads/text2sql/eval.sbatch:
# validation, arcwise, hard; T=0, 1 sample per task). Idempotent: a step whose eval dir already has queue.log is skipped,
# so a requeued (preempted) helper or a rerun never queues the same eval twice. Runs in a hopper-cpu job:
#   sbatch -p hopper-cpu --qos=low -c 2 --mem=16G -t 2-0 -J nemo-rl-t2sql-evals \
#       -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out evals.sh <b200-queue waiter id> ["20 40 60 80 100"]
# Per step: complete_ckpt.py makes stable-bench-runs/nemo-rl/t2sql/hf/<run>-checkpoint-N from
# <run>/checkpoints/step_N/policy/weights/model/consolidated, then eval.sbatch writes
# stable-bench-runs/trl-moe-ep/t2sql-eval/<run>-checkpoint-N/summary.json.
HERE=$(cd "$(dirname "$0")" && pwd)
T2=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/text2sql
PY=/fsx/amine_dirhoussi/envs/bench-trl-ep/bin/python
MERGE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/prime-rl/merge_lora.py  # streaming PEFT merge, shard by shard
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
STATE=/fsx/amine_dirhoussi/stable-bench-runs/queue/state/$1.jid; STEPS=${2:-20 40 60 80 100}
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql/data
until [ -s "$STATE" ]; do sleep 120; done
J=$(cat "$STATE"); RUN=${RUN_PREFIX:-nemo-rl-t2sql-Qwen3.5-35B-A3B}-$J
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/t2sql/$RUN
echo "$(date -u +%FT%TZ) run job $J ($RUN_DIR)"
SPECS=""; for e in validation hard arcwise; do SPECS="$SPECS eval-$e=$D/$e.jsonl:1:0"; done
declare -A done; n=$(echo $STEPS | wc -w)
while [ ${#done[@]} -lt $n ]; do
  for N in $STEPS; do
    [ -n "${done[$N]:-}" ] && continue
    OUT=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/t2sql-eval/$RUN-checkpoint-$N
    [ -f $OUT/queue.log ] && { done[$N]=1; continue; }
    C=$RUN_DIR/checkpoints/step_$N
    # NeMo-RL writes the step dir under a tmp name and renames it when complete. Full fine-tune: consolidated HF
    # shards; LoRA: a PEFT adapter (adapter_model.safetensors), folded into the base with the streaming merger.
    HF=/fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/t2sql/hf/$RUN-checkpoint-$N
    ADAPTER=$(find $C/policy/weights -name adapter_model.safetensors 2>/dev/null | head -1)
    if [ -n "$ADAPTER" ]; then
      if [ ! -f $HF/complete.json ]; then
        sleep 120
        echo "$(date -u +%FT%TZ) step $N: merging adapter $(dirname $ADAPTER) -> $HF"
        $PY $MERGE $(dirname $ADAPTER) $HF Qwen/Qwen3.5-35B-A3B > $HF.merge.json 2>&1 && cp $HF.merge.json $HF/complete.json \
          || { echo "merge failed for step $N: $(tail -3 $HF.merge.json)"; sleep 300; continue; }
      fi
    else
      [ -d $C/policy/weights/model/consolidated ] || continue
      if [ ! -f $HF/complete.json ]; then
        sleep 120  # let the consolidated shards finish writing
        echo "$(date -u +%FT%TZ) step $N: completing $C -> $HF"
        $PY $HERE/complete_ckpt.py $C/policy/weights/model/consolidated $HF || { echo "complete failed for step $N"; sleep 300; continue; }
      fi
    fi
    mkdir -p $OUT
    echo "$(date -u +%FT%TZ) step $N ready, queueing eval -> $OUT"
    (cd /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench && PRIO_TAG=PRIO_TRAIN nohup /fsx/amine_dirhoussi/bin/b200-queue \
      -J t2sql-eval --gres=gpu:4 -t 4:00:00 --export=ALL,MODEL=$HF,OUT=$OUT,NGPU=4,SPECS="$SPECS" \
      $T2/eval.sbatch > $OUT/queue.log 2>&1 &)
    done[$N]=1
  done
  [ -n "${ended:-}" ] && { echo "$(date -u +%FT%TZ) run $J ended; queued: ${!done[*]}"; break; }
  squeue -h -j "$J" | grep -q . || ended=1
  sleep 120
done
wait
