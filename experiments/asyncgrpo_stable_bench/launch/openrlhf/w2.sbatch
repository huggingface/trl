#!/bin/bash
# W2 agentic (TermGrade, Qwen3.5-9B) with OpenRLHF on one 8 x B200 node: 4 vLLM engines + 4 DeepSpeed ranks, async,
# episodes on the OpenEnv Harbor server TRL/verl use, local Apptainer boxes (as launch/verl/w2.sbatch).
#   PRIO_TAG=PRIO_TRAIN b200-queue --gres=gpu:8 --exclusive -t 16:00:00 launch/openrlhf/w2.sbatch
# Knobs: TAG, MAX_STEPS (60), SAVE_STEPS (20), ZERO_STAGE (3), ADAM_OFFLOAD, HARBOR_SHARDS (8), AGENT_TIMEOUT, STEP_LIMIT,
# RESUME_DIR (an earlier run dir to continue from its DeepSpeed checkpoint: --ckpt.load_enable).
#SBATCH --job-name=w2local-orlhf
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/logs/%x-%j.out
set -uo pipefail
TAG=${TAG:-shared}
RUN_ID=openrlhf-w2-M-${TAG}-${SLURM_JOB_ID}
RUN_DIR=${RESUME_DIR:-/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/$RUN_ID}
mkdir -p "$RUN_DIR/episodes"
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
MODEL=$(ls -d $HF_HUB_CACHE/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)
W2_VENV=/fsx/amine_dirhoussi/envs/bench-trl-w2
OPENENV=/fsx/amine_dirhoussi/stable-bench-src/openenv
SPLIT_DIR=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train
export W2_PYTHON=$W2_VENV/bin/python W2_PYTHONPATH=$OPENENV/envs:$BENCH/workloads/agentic W2_SANDBOX=singularity
export W2_EPISODES_DIR=$RUN_DIR/episodes AGENT_TIMEOUT=${AGENT_TIMEOUT:-600} STEP_LIMIT=${STEP_LIMIT:-30} MAX_MODEL_LEN=65536
export OPENENV_HARBOR_TRIALS_DIR=/scratch/$USER/w2-trials-$SLURM_JOB_ID; mkdir -p "$OPENENV_HARBOR_TRIALS_DIR"
HARBOR_SHARDS=${HARBOR_SHARDS:-8}
HPORT=$((62000 + (SLURM_JOB_ID % 90) * 20)); STUB_PORT=$HPORT
HARBOR_PORTS=(); for i in $(seq 1 $HARBOR_SHARDS); do HARBOR_PORTS+=($((HPORT + 2 * i))); done
export HARBOR_URL=http://localhost:${HARBOR_PORTS[0]}
export HARBOR_URLS=$(printf 'http://localhost:%s,' "${HARBOR_PORTS[@]}" | sed 's/,$//')
# Ray actors (the vLLM rollout actors run the agent executor) inherit the environment of `ray start`: export first.
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/openrlhf/common.sh
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh > "$RUN_DIR/apptainer-env.log" 2>&1
BOX_DIR=/scratch/$USER/w2-boxes-$SLURM_JOB_ID; mkdir -p "$BOX_DIR/tmp" "$BOX_DIR/apptainer-cache" "$BOX_DIR/apptainer-tmp"
HARBOR_ENV=(LD_LIBRARY_PATH=$AP_LIBS APPTAINER_BIND=/etc/resolv.conf TMPDIR=$BOX_DIR/tmp
            APPTAINER_CACHEDIR=$BOX_DIR/apptainer-cache APPTAINER_TMPDIR=$BOX_DIR/apptainer-tmp)
HARBOR_PIDS=(); STUB_PID=""
tree_pids() { ps -u $USER -o pid=,ppid= | awk -v r=$1 '{p[$1]=$2} END {do {c=0; for (x in p) if (p[x] in s || p[x]==r) if (!(x in s)) {s[x]=1; c++}} while (c); for (x in s) print x}'; }
w2_cleanup() {
    BOX_PIDS=""; for p in "${HARBOR_PIDS[@]}"; do BOX_PIDS="$BOX_PIDS $(tree_pids $p)"; done
    kill "${HARBOR_PIDS[@]}" $STUB_PID $GPU_SAMPLER 2>/dev/null; sleep 5
    kill ${BOX_PIDS:-} 2>/dev/null; pkill -u $USER -f "$BOX_DIR" 2>/dev/null; sleep 5
    kill -9 ${BOX_PIDS:-} 2>/dev/null; pkill -9 -u $USER -f "$BOX_DIR" 2>/dev/null
    rm -rf "$BOX_DIR"
    tar czf "$RUN_DIR/trials-$SLURM_JOB_ID.tgz" -C "$OPENENV_HARBOR_TRIALS_DIR" --exclude='*/agent/*.cast' . 2>/dev/null && rm -rf "$OPENENV_HARBOR_TRIALS_DIR"
}
trap w2_cleanup EXIT

DATA=$RUN_DIR/data/train.parquet; mkdir -p "$RUN_DIR/data"
python "$BENCH/configs/openrlhf/w2_agent.py" data "$SPLIT_DIR" "$DATA" $((8 * ${MAX_STEPS:-60})) || exit 1
source "$BENCH/configs/openrlhf/w2-M.sh"
cp "$BENCH/configs/openrlhf/w2_agent.py" "$BENCH/configs/openrlhf/w2-M.sh" "$RUN_DIR/"
printf '%s\n' "${W1_ARGS[@]}" "$@" > "$RUN_DIR/args.txt"
echo "$SLURM_JOB_ID $(hostname)" >> "$RUN_DIR/jobs"

python "$BENCH/configs/openrlhf/w2_agent.py" stub $STUB_PORT > "$RUN_DIR/stub.log" 2>&1 &
STUB_PID=$!
until curl -sf "http://127.0.0.1:$STUB_PORT/v1/models" > /dev/null; do kill -0 $STUB_PID 2>/dev/null || { echo "stub died"; exit 1; }; sleep 2; done
for i in "${!HARBOR_PORTS[@]}"; do
    env "${HARBOR_ENV[@]}" PATH=$W2_VENV/bin:$PATH PYTHONPATH=$W2_PYTHONPATH MAX_CONCURRENT_ENVS=72 \
        $W2_VENV/bin/python "$BENCH/workloads/agentic/harbor_serve.py" harbor serve \
        --dataset "$SPLIT_DIR" --llm-url "http://127.0.0.1:$STUB_PORT" --model policy \
        --host 127.0.0.1 --port ${HARBOR_PORTS[$i]} --capture-port $((HARBOR_PORTS[$i] + 1)) --expose direct \
        --max-output-tokens 8192 --no-rollouts --no-run-history > "$RUN_DIR/harbor-$SLURM_JOB_ID-$i.log" 2>&1 &
    HARBOR_PIDS+=($!)
done
for i in "${!HARBOR_PORTS[@]}"; do
    until curl -sf "http://localhost:${HARBOR_PORTS[$i]}/health" > /dev/null; do
        kill -0 ${HARBOR_PIDS[$i]} 2>/dev/null || { echo "harbor server $i died"; tail -n 30 "$RUN_DIR"/harbor-$SLURM_JOB_ID-$i.log; exit 1; }
        sleep 5
    done
done
kill $STUB_PID 2>/dev/null; STUB_PID=""
echo "harbor ready $(date -u +%FT%TZ) shards=$HARBOR_SHARDS"

python3 -m openrlhf.cli.train_ppo_ray "${W1_ARGS[@]}" "$@" &
TRAIN=$!
LOG=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *StdOut=//p')
( python "$BENCH/launch/openrlhf/w2_watch.py" "$RUN_DIR" "$LOG" > "$RUN_DIR/watchdog-$SLURM_JOB_ID.log" 2>&1 \
  || { tail -1 "$RUN_DIR/watchdog-$SLURM_JOB_ID.log"; pkill -KILL -P $TRAIN; kill -KILL $TRAIN; } ) &
(while sleep 60; do
     [ $(( $(date +%s) - $(stat -c %Y "$LOG") )) -gt $(( ${STALL_MIN:-45} * 60 )) ] || continue
     echo "WATCHDOG: no log output for ${STALL_MIN:-45} min, killing the trainer" | tee "$RUN_DIR/WATCHDOG"
     pkill -KILL -P $TRAIN; kill -KILL $TRAIN; break
 done) &
wait $TRAIN
RC=$?
[ -f "$RUN_DIR/WATCHDOG" ] && RC=124
echo "JOB_END $(date +%s.%N) rc=$RC" >> "$RUN_DIR/job_times.txt"
exit $RC
