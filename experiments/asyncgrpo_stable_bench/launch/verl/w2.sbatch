#!/bin/bash
#SBATCH --job-name=w2local-verl
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/%x-%j.out
# W2 agentic with verl (V1 separate_async, 4 vLLM + 4 FSDP2 GPUs) on the OpenEnv Harbor server TRL uses, local
# Apptainer boxes (as launch/trl/w2.sbatch SANDBOX=local). verl's agent loop is configs/verl/w2_agent_loop.py.
#   PRIO_TAG=PRIO_TRAIN TAG=shared b200-queue -J w2local-verl launch/verl/w2.sbatch
# Smoke checks run inside every job (launch/verl/w2_watch.py): a failed check stops training with WATCHDOG in the run dir.
# Knobs: TAG, MAX_STEPS, SAVE_STEPS, HARBOR_SHARDS (8), TERMINUS_MAX_TURNS (1), AGENT_TIMEOUT (600), STEP_LIMIT (30).
# The job name must not start with trl-w2 (sweep.sh); local boxes are cleaned here, no HF sweep needed.
set -uo pipefail
TAG=${TAG:-shared}
CONFIG=${CONFIG:-w2-M}   # w2-moe: Qwen3.5-35B-A3B + LoRA (checkpoints merged on this node by t2sql_merge.sh)
SIZE=M; [ "$CONFIG" = w2-moe ] && SIZE=moe
OVR=()
[ -n "${MAX_STEPS:-}" ] && OVR+=(trainer.total_training_steps=$MAX_STEPS)
[ -n "${SAVE_STEPS:-}" ] && OVR+=(trainer.save_freq=$SAVE_STEPS)
# No PYTORCH_CUDA_ALLOC_CONF=expandable_segments here (W1-M sets it): Ray workers inherit it, and verl's weight sync hands
# the checkpoint-engine buffer to vLLM by CUDA IPC; with expandable segments torch shares it through pidfd_getfd, which the
# node refuses between unrelated processes ("pidfd_getfd: Operation not permitted", job 371788). verl's vLLM server turns
# expandable segments on for itself anyway.
# v0.9.1 separate_async + hybrid_engine=false needs WARN (envs/verl/NOTES.md "As shipped").
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-WARN}
RUN_ID=verl-w2-$SIZE-$TAG-$SLURM_JOB_ID
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/verl/$RUN_ID
mkdir -p "$RUN_DIR/episodes"
[ -s "$RUN_DIR/metrics.jsonl" ] && mv "$RUN_DIR/metrics.jsonl" "$RUN_DIR/metrics.$(date +%s).jsonl"
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
W2_VENV=/fsx/amine_dirhoussi/envs/bench-trl-w2
OPENENV=/fsx/amine_dirhoussi/stable-bench-src/openenv
SPLIT_DIR=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train

# What the agent-loop workers (Ray actors, which inherit this environment from `ray start`) need.
export PYTHONPATH=$BENCH/configs/verl
export W2_PYTHON=$W2_VENV/bin/python W2_PYTHONPATH=$OPENENV/envs:$BENCH/workloads/agentic W2_SANDBOX=singularity
export W2_EPISODES_DIR=$RUN_DIR/episodes
export AGENT_TIMEOUT=${AGENT_TIMEOUT:-600} STEP_LIMIT=${STEP_LIMIT:-30} MAX_MODEL_LEN=65536
export OPENENV_HARBOR_TRIALS_DIR=/scratch/$USER/w2-trials-$SLURM_JOB_ID
mkdir -p "$OPENENV_HARBOR_TRIALS_DIR"
HARBOR_SHARDS=${HARBOR_SHARDS:-8}; TERMINUS_MAX_TURNS=${TERMINUS_MAX_TURNS:-1}
# Above Linux's ephemeral range (32768-60999) and outside the Ray ports of common.sh.
HPORT=$((62000 + (SLURM_JOB_ID % 90) * 20))
STUB_PORT=$HPORT
HARBOR_PORTS=(); for i in $(seq 1 $HARBOR_SHARDS); do HARBOR_PORTS+=($((HPORT + 2 * i))); done
export HARBOR_URL=http://localhost:${HARBOR_PORTS[0]}
export HARBOR_URLS=$(printf 'http://localhost:%s,' "${HARBOR_PORTS[@]}" | sed 's/,$//')

# Local boxes: the unprivileged Apptainer of launch/trl/w2.sbatch, same env for the Harbor servers.
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh > "$RUN_DIR/apptainer-env.log" 2>&1
BOX_DIR=/scratch/$USER/w2-boxes-$SLURM_JOB_ID
mkdir -p "$BOX_DIR/tmp" "$BOX_DIR/apptainer-cache" "$BOX_DIR/apptainer-tmp"
HARBOR_ENV=(LD_LIBRARY_PATH=$AP_LIBS APPTAINER_BIND=/etc/resolv.conf TMPDIR=$BOX_DIR/tmp
            APPTAINER_CACHEDIR=$BOX_DIR/apptainer-cache APPTAINER_TMPDIR=$BOX_DIR/apptainer-tmp)
apptainer --version > "$RUN_DIR/apptainer_version" 2>&1

source $BENCH/launch/verl/common.sh   # venv, Ray head, gpu.csv, `cleanup`; cd's into the verl checkout

HARBOR_PIDS=(); STUB_PID=""
tree_pids() { ps -u $USER -o pid=,ppid= | awk -v r=$1 '{p[$1]=$2} END {do {c=0; for (x in p) if (p[x] in s || p[x]==r) if (!(x in s)) {s[x]=1; c++}} while (c); for (x in s) print x}'; }
w2_cleanup() {
    # Local boxes are processes of this job: stop the servers and any box they left, then remove scratch and caches
    # (launch/trl/w2.sbatch SANDBOX=local).
    BOX_PIDS=""; for p in "${HARBOR_PIDS[@]}"; do BOX_PIDS="$BOX_PIDS $(tree_pids $p)"; done
    kill "${HARBOR_PIDS[@]}" $STUB_PID 2>/dev/null
    cleanup
    sleep 5
    kill ${BOX_PIDS:-} 2>/dev/null; pkill -u $USER -f "$BOX_DIR" 2>/dev/null; sleep 5
    kill -9 ${BOX_PIDS:-} 2>/dev/null; pkill -9 -u $USER -f "$BOX_DIR" 2>/dev/null
    echo "box processes at cleanup: $(echo ${BOX_PIDS:-} | wc -w); left after: $(pgrep -u $USER -fc "$BOX_DIR")" > "$RUN_DIR/sweep.log"
    rm -rf "$BOX_DIR"
    tar czf "$RUN_DIR/trials.tgz" -C "$OPENENV_HARBOR_TRIALS_DIR" --exclude='*/agent/*.cast' . 2>/dev/null && rm -rf "$OPENENV_HARBOR_TRIALS_DIR"
    date +%s.%N > "$RUN_DIR/job_end"
}
trap w2_cleanup EXIT

DATA=$RUN_DIR/data
mkdir -p "$DATA"
python "$BENCH/configs/verl/w2_agent_loop.py" data "$SPLIT_DIR" "$DATA/train.parquet" || exit 1
cp "$BENCH/configs/verl/$CONFIG.yaml" "$RUN_DIR/config.yaml"; cp "$BENCH/configs/verl/w2_agent_loop.py" "$RUN_DIR/"; echo $CONFIG > "$RUN_DIR/config_name"
git -C "$SRC" rev-parse HEAD > "$RUN_DIR/verl_sha"
echo "$SLURM_JOB_NODELIST" > "$RUN_DIR/nodes"
env | grep -E '^(TAG|MAX_STEPS|SAVE_STEPS|HARBOR_SHARDS|TERMINUS_MAX_TURNS|AGENT_TIMEOUT|STEP_LIMIT|MAX_MODEL_LEN|W2_SANDBOX)=' > "$RUN_DIR/launch.env"
echo "HARBOR_SHARDS=$HARBOR_SHARDS TERMINUS_MAX_TURNS=$TERMINUS_MAX_TURNS" >> "$RUN_DIR/launch.env"

# The Harbor server checks at startup that its engine answers; verl's servers only exist once training starts, so a
# stand-in answers that check (every rollout then names its own engine: the agent loop's proxy).
python "$BENCH/configs/verl/w2_agent_loop.py" stub $STUB_PORT > "$RUN_DIR/stub.log" 2>&1 &
STUB_PID=$!
until curl -sf "http://127.0.0.1:$STUB_PORT/v1/models" > /dev/null; do kill -0 $STUB_PID 2>/dev/null || { echo "stub died"; cat "$RUN_DIR/stub.log"; exit 1; }; sleep 2; done
CLI=($W2_VENV/bin/openenv)
[ "$TERMINUS_MAX_TURNS" = 1 ] && CLI=($W2_VENV/bin/python "$BENCH/workloads/agentic/harbor_serve.py")
for i in "${!HARBOR_PORTS[@]}"; do
    log=$RUN_DIR/harbor.log; [ $i -gt 0 ] && log=$RUN_DIR/harbor-$i.log
    env "${HARBOR_ENV[@]}" PATH=$W2_VENV/bin:$PATH PYTHONPATH=$W2_PYTHONPATH MAX_CONCURRENT_ENVS=72 "${CLI[@]}" harbor serve \
        --dataset "$SPLIT_DIR" --llm-url "http://127.0.0.1:$STUB_PORT" --model policy \
        --host 127.0.0.1 --port ${HARBOR_PORTS[$i]} --capture-port $((HARBOR_PORTS[$i] + 1)) --expose direct \
        --max-output-tokens 8192 --no-rollouts --no-run-history > "$log" 2>&1 &
    HARBOR_PIDS+=($!)
done
for i in "${!HARBOR_PORTS[@]}"; do
    until curl -sf "http://localhost:${HARBOR_PORTS[$i]}/health" > /dev/null; do
        kill -0 ${HARBOR_PIDS[$i]} 2>/dev/null || { echo "harbor server $i died during startup"; tail -n 30 "$RUN_DIR"/harbor*.log; exit 1; }
        sleep 5
    done
    csv=$RUN_DIR/boxes.csv; [ $i -gt 0 ] && csv=$RUN_DIR/boxes-$i.csv
    $W2_VENV/bin/python "$BENCH/workloads/agentic/box_usage.py" ${HARBOR_PIDS[$i]} 5 > "$csv" &
done
kill $STUB_PID 2>/dev/null; STUB_PID=""
echo "harbor ready at $(date -u +%FT%TZ), shards=$HARBOR_SHARDS terminus_max_turns=$TERMINUS_MAX_TURNS" | tee "$RUN_DIR/harbor_ready"
( while sleep 60; do echo "$(date +%s) $(ps -o pid=,pcpu=,cputime= -p "$(IFS=,; echo "${HARBOR_PIDS[*]}")" | tr '\n' ' ')"; done ) > "$RUN_DIR/harbor_cpu.txt" &

# In-job smoke checks (episodes, first steps, first checkpoint): on a failure the watchdog writes WATCHDOG and stops training.
( python "$BENCH/launch/verl/w2_watch.py" "$RUN_DIR" "${SAVE_STEPS:-20}" > "$RUN_DIR/watchdog.log" 2>&1 \
  || { cat "$RUN_DIR/watchdog.log" | tail -1; pkill -u $USER -f verl.trainer.main_ppo; } ) &

MERGE_PID=""
if [ "$SIZE" = moe ]; then
    mkdir -p /fsx/amine_dirhoussi/stable-bench-runs/verl/merged
    bash "$BENCH/launch/verl/t2sql_merge.sh" "$RUN_DIR" loop > "$RUN_DIR/merge.log" 2>&1 &
    MERGE_PID=$!
fi
python -m verl.trainer.main_ppo --config-path="$BENCH/configs/verl" --config-name=$CONFIG \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/train.parquet" \
  trainer.experiment_name="$RUN_ID" \
  trainer.default_local_dir="$RUN_DIR/ckpt" \
  "${OVR[@]}" \
  "$@" 2>&1 | perl -MTime::HiRes=time -ne '$|=1; printf "%.3f %s", time, $_' | tee -a "$RUN_DIR/stdout.log"
RC=${PIPESTATUS[0]}
echo "EXIT $RC"
[ -n "$MERGE_PID" ] && { kill $MERGE_PID 2>/dev/null; bash "$BENCH/launch/verl/t2sql_merge.sh" "$RUN_DIR" once >> "$RUN_DIR/merge.log" 2>&1; }
w2_cleanup; trap - EXIT
python "$BENCH/collect/verl.py" "$RUN_DIR" --job-log "/fsx/amine_dirhoussi/stable-bench-runs/verl/logs/w2local-verl-$SLURM_JOB_ID.out" --status $([ $RC = 0 ] && echo completed || echo failed) || true
python "$BENCH/collect/verl_w2.py" "$RUN_DIR" || true
exit $RC
