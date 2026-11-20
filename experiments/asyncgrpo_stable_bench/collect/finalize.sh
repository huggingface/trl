#!/bin/bash
# Post-run step for one finished run: lib-specific prep, collect/<lib>.py in a hopper-cpu job, then one 1-GPU
# workloads/math/eval.sbatch job per checkpoint, appending to results/raw/<run_id>/eval.jsonl.
#   collect/finalize.sh [--dry-run] <lib> <run_dir>
# Run it from a background shell: it waits for the collector job and for every eval job (b200-queue waits).
# Env: STATUS (passed to the collector as --status), EVAL_NODE (pin evals to this node instead of picking one),
#      EVAL_PY (python for eval.sbatch, defaults to the TRL bench venv), SKIP_COLLECT=1, SKIP_EVAL=1.
# Never run it on a live run: the collector would write a partial steps.jsonl and eval would miss checkpoints.
set -uo pipefail

DRY=0
[ "${1:-}" = --dry-run ] && { DRY=1; shift; }
[ $# -eq 2 ] || { echo "usage: $0 [--dry-run] <lib> <run_dir>" >&2; exit 2; }
LIB=$1; RUN_DIR=$(readlink -f "$2"); RUN_ID=$(basename "$RUN_DIR")
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
RUNS=/fsx/amine_dirhoussi/stable-bench-runs
OUT=$BENCH/results/raw/$RUN_ID/eval.jsonl
LOGS=$RUNS/$LIB/logs
[ -d "$RUN_DIR" ] || { echo "no run dir $RUN_DIR" >&2; exit 1; }

run() {  # print, then run unless --dry-run
    echo "+ $*"
    [ $DRY = 1 ] || "$@"
}

# 1. Lib-specific prep: expose every HF export as <run_dir>/checkpoint-<optimizer step>, the layout eval.sbatch reads.
planned=()  # steps linked by this call (in --dry-run the links don't exist yet, but their evals are still listed)
link() {  # link <target relative to run dir> <step>
    [ -e "$RUN_DIR/checkpoint-$2" ] || { run ln -sfn "$1" "$RUN_DIR/checkpoint-$2"; planned+=("$2"); }
}
case $LIB in
    trl) ;;  # writes checkpoint-<step> itself
    miles)  # exports are named by rollout id, step = rollout_id + 1
        for r in 49 99 149 199; do [ -d "$RUN_DIR/hf/step_$r" ] && link "hf/step_$r" $((r + 1)); done ;;
    prime-rl)
        for d in "$RUN_DIR"/hf/step_*; do [ -d "$d" ] && link "hf/${d##*/}" "${d##*_}"; done ;;
    skyrl)
        for d in "$RUN_DIR"/global_step_*/policy; do [ -d "$d" ] && s=${d%/policy} && link "${s##*/}/policy" "${s##*_}"; done ;;
    verl)  # a verl global step spans parameter_sync_step optimizer steps (recommended arm: 4); link by optimizer step
        k=$(grep -oE '^\s*parameter_sync_step:\s*[0-9]+' "$RUN_DIR/config.yaml" 2>/dev/null | grep -oE '[0-9]+$' | head -1)
        for d in "$RUN_DIR"/ckpt/global_step_*/actor/huggingface; do
            [ -d "$d" ] && s=${d%/actor/huggingface} && link "ckpt/${s##*/}/actor/huggingface" "$(( ${s##*_} * ${k:-1} ))"
        done ;;
    openrlhf)
        for d in "$RUN_DIR"/ckpt/global_step*_hf; do [ -d "$d" ] && s=${d##*global_step} && link "ckpt/${d##*/}" "${s%_hf}"; done ;;
    *) echo "unknown lib $LIB" >&2; exit 2 ;;
esac

# 2. Collector, in a hopper-cpu job with the lib's venv. Arguments differ per collector (see each one's header).
PY=/fsx/amine_dirhoussi/envs/bench-$LIB/bin/python
ARGS=("$RUN_DIR")
case $LIB in
    miles) PY="$(command -v uv) run --no-project --python 3.12 python" ;;  # stdlib only; bench-miles has no python
    prime-rl | skyrl) ARGS=("$RUN_ID") ;;  # they take the run id and find the run under their RUNS dir
    verl) ARGS+=(--job-log "$LOGS/verl-w1-${RUN_ID##*-}.out") ;;
esac
[ -n "${STATUS:-}" ] && ARGS+=(--status "$STATUS")
if [ "${SKIP_COLLECT:-0}" != 1 ]; then
    run mkdir -p "$LOGS"
    run sbatch --wait -p hopper-cpu --qos=low -c 4 --mem=32G -t 1:00:00 -J "collect-$LIB" -o "$LOGS/%x-%j.out" \
        --wrap "cd $BENCH && $PY collect/$LIB.py ${ARGS[*]}" || { echo "collector failed, see $LOGS" >&2; exit 1; }
fi

# 3. One eval job per run for the checkpoints not yet in eval.jsonl, linked into one batch dir
#    (eval.sbatch evaluates every checkpoint-* of the dir it is given, one per GPU).
[ "${SKIP_EVAL:-0}" = 1 ] && exit 0
done_steps=$(grep -o '"step": [0-9]*' "$OUT" 2>/dev/null | awk '{print $2}')
node=${EVAL_NODE:-}
if [ -z "$node" ]; then  # the node of mine on hopper-b200 with the most free GPUs (it may have none: the job waits)
    best=-1
    for n in $(squeue -u "$USER" -p hopper-b200 -h -t R -o %N | xargs -r -n1 scontrol show hostnames | sort -u); do
        alloc=$(scontrol show node "$n" | grep -o 'AllocTRES=[^ ]*' | grep -o 'gres/gpu=[0-9]*' | cut -d= -f2)
        free=$((8 - ${alloc:-0}))
        [ $free -gt $best ] && { best=$free; node=$n; }
    done
    [ "$best" -gt 0 ] || node=""  # every node of mine is full (exclusive runs): don't pin, let b200-queue find a slot
fi
echo "eval node: ${node:-none of mine running, b200-queue waits for a free slot under the cap}"
# All pending checkpoints of the run go into one job (one GPU each, eval.sbatch runs them in parallel): the node guard
# counts every pending job as a node, so one job per checkpoint would fill the cap with tiny jobs.
pids=()
steps=$( (ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sed 's/.*-//'; printf '%s\n' "${planned[@]}") | grep . | sort -nu)
dir=$RUN_DIR/eval-split/batch-$(date -u +%Y%m%dT%H%M%S)
n=0
for step in $steps; do
    if grep -qx "$step" <<<"$done_steps"; then echo "step $step already in $OUT"; continue; fi
    run mkdir -p "$dir"
    run ln -sfn "$RUN_DIR/checkpoint-$step" "$dir/checkpoint-$step"
    n=$((n + 1))
done
if [ $n -gt 0 ]; then
    g=$(( n > 8 ? 8 : n ))
    if [ $DRY = 1 ]; then
        echo "+ /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:$g ${node:+-w $node} $BENCH/workloads/math/eval.sbatch $dir $OUT ${EVAL_PY:-}"
    else
        /fsx/amine_dirhoussi/bin/b200-queue --gres=gpu:$g ${node:+-w "$node"} "$BENCH/workloads/math/eval.sbatch" \
            "$dir" "$OUT" ${EVAL_PY:-} &
        pids+=($!)
    fi
fi
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $DRY = 1 ] || echo "evals done (rc=$rc): $OUT"
exit $rc
