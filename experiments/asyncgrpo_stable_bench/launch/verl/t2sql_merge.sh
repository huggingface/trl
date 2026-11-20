#!/bin/bash
# Merge every finished LoRA checkpoint of a verl text-to-SQL run into a full HF model for the shared eval
# (workloads/text2sql/eval.sbatch), on the CPU of the training node. Idempotent: skips merged steps.
#   t2sql_merge.sh <run_dir> loop    # inside the training job: poll until killed
#   t2sql_merge.sh <run_dir> once    # one pass (after training)
# Output: stable-bench-runs/verl/merged/<run id>-checkpoint-N/ (merge_check.json last) + adapter-N/ in the run dir.
RUN_DIR=$1; MODE=${2:-once}
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
RUN_ID=$(basename "$RUN_DIR")
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
while true; do
    latest=$(cat "$RUN_DIR/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    for d in "$RUN_DIR"/ckpt/global_step_*; do
        [ -d "$d/actor" ] || continue
        N=${d##*_}
        [ "$N" -le "$latest" ] || continue
        out=/fsx/amine_dirhoussi/stable-bench-runs/verl/merged/$RUN_ID-checkpoint-$N
        [ -f "$out/merge_check.json" ] || [ -f "$RUN_DIR/merge-$N.failed" ] && continue
        echo "$(date -u +%FT%TZ) merging step $N"
        /fsx/amine_dirhoussi/envs/bench-verl/bin/python "$BENCH/launch/verl/t2sql_export.py" "$d/actor" "$RUN_DIR/adapter-$N" \
            > "$RUN_DIR/merge-$N.log" 2>&1 \
        && /fsx/amine_dirhoussi/envs/bench-trl-ep/bin/python "$BENCH/launch/trl/merge_lora.py" \
            "$RUN_DIR/adapter-$N" "$out" Qwen/Qwen3.5-35B-A3B >> "$RUN_DIR/merge-$N.log" 2>&1
        rc=$?; echo "$(date -u +%FT%TZ) step $N merge exit $rc -> $out"
        [ $rc = 0 ] || touch "$RUN_DIR/merge-$N.failed"
    done
    [ "$MODE" = loop ] || break
    sleep 120
done
