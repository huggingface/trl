#!/bin/bash
# W2 (agentic, TermGrade on HF Sandboxes, terminus-2) on one node: vLLM on the first NVLLM GPUs, the OpenEnv Harbor
# server on the CPU (terminus-2 runs inside it and drives each sandbox over the HF Sandbox API), and either the
# base-rate probe or the AsyncGRPO trainer (FSDP2) on the remaining GPUs.
#   probe:      STAGE=probe PROBE_SIZES="M S" NVLLM=4 SPLIT=probe b200-queue --gres=gpu:4 launch/trl/w2.sbatch
#   smoke:      SIZE=M NVLLM=2 NTRAIN=2 MAX_STEPS=3 TAG=smoke b200-queue --gres=gpu:4 launch/trl/w2.sbatch
#   timed run:  SIZE=M b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/trl/w2.sbatch
#   local (default for training): SIZE=M b200-queue -J w2local-trl --nodes=1 --gres=gpu:8 --exclusive launch/trl/w2.sbatch
#               (same run on local containers: Harbor's `singularity` env on an unprivileged Apptainer on this node,
#               tasks from harbor-local-sif/, run id trl-w2-M-local-<jobid>; see configs/trl/EQUIVALENCE.md W2). The job
#               name must not start with trl-w2: sweep.sh skips the HF sandbox sweep while another trl-w2* job runs.
# Env knobs: STAGE (train | probe), SIZE (S | M), PROBE_SIZES (probe: e.g. "M S"), TAG, NVLLM (4), NTRAIN (4), SPLIT (train; a dir under
# stable-bench-src/termgrade/harbor), MAX_STEPS, SAVE_STEPS, CONCURRENCY (in-flight episodes = live sandboxes, 64),
# AGENT_TIMEOUT (600), STEP_LIMIT (30), MAX_MODEL_LEN (65536), SANDBOX (hf | local), probe only: SAMPLES (4), N_TASKS (0 = all).
# HARBOR_SHARDS (1): Harbor servers on the same tasks, rollouts spread round-robin (termgrade_env.py HARBOR_URLS), so no
# single server event loop runs every agent (envs/trl/W2_STEP_TIME.md cause 1). TERMINUS_MAX_TURNS (0): 1 passes STEP_LIMIT
# to terminus-2 as its own max_turns (workloads/agentic/harbor_serve.py), so an episode stops at the call budget instead
# of re-asking until the agent timeout (cause 4).
#SBATCH -J trl-w2
#SBATCH --nodes=1
#SBATCH -t 24:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl/jobs/%x-%j.out
set -uo pipefail
JOB_START=$(date +%s.%N)
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
VLLM_VENV=/fsx/amine_dirhoussi/envs/bench-trl       # the W1 server stack, unchanged
W2_VENV=/fsx/amine_dirhoussi/envs/bench-trl-w2      # trainer + OpenEnv Harbor server (mcp<2)
OPENENV=/fsx/amine_dirhoussi/stable-bench-src/openenv
# Training defaults to local containers (user, 2026-11-11: TRL W2 is benchmarked on local Apptainer boxes on the training node).
# Held-out evals (STAGE=probe) stay on HF Sandboxes so every checkpoint, TRL or prime-rl, is scored on the same backend.
SANDBOX=${SANDBOX:-$([ "${STAGE:-train}" = probe ] && echo hf || echo local)}
TAG_DEFAULT=shared; [ "$SANDBOX" = local ] && TAG_DEFAULT=local
STAGE=${STAGE:-train}; SIZE=${SIZE:-M}; TAG=${TAG:-$TAG_DEFAULT}; NVLLM=${NVLLM:-4}; NTRAIN=${NTRAIN:-4}
# Run-shape values decided from the probe (SPLIT, GRAD_ACCUM, ...) live in configs/trl/w2-$SIZE.env, read at job start;
# a value already set in the submitting environment wins (smoke runs: MAX_STEPS=3 ...).
if [ "$STAGE" = train ] && [ -f "$BENCH/configs/trl/w2-$SIZE.env" ]; then
    while IFS='=' read -r k v; do [ -z "${!k+x}" ] && export "$k=$v"; done < <(grep -E '^[A-Z_]+=' "$BENCH/configs/trl/w2-$SIZE.env")
fi
SPLIT=${SPLIT:-train}
export CONCURRENCY=${CONCURRENCY:-64} AGENT_TIMEOUT=${AGENT_TIMEOUT:-600} STEP_LIMIT=${STEP_LIMIT:-30}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
declare -A MODELS=([S]=Qwen/Qwen3.5-2B [M]=Qwen/Qwen3.5-9B [L]=Qwen/Qwen3.8-27B)
export MODEL=${MODELS[$SIZE]}
if [ "$STAGE" = probe ]; then export RUN_ID=trl-w2probe-$SIZE-$SPLIT-$SLURM_JOB_ID; else export RUN_ID=trl-w2-$SIZE-$TAG-$SLURM_JOB_ID; fi
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/trl/$RUN_ID
mkdir -p "$RUN_DIR"
echo "$JOB_START" > "$RUN_DIR/job_start"
export SPLIT_DIR=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor/$SPLIT
if [ "$SANDBOX" = local ]; then
    # Same tasks with docker_image pointing at the .sif of the same base image (workloads/agentic/NATIVE_SANDBOXES.md).
    export SPLIT_DIR=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/$SPLIT W2_SANDBOX=singularity
fi
# A job queued before its split was prepared waits for it (up to 90 min) instead of failing.
for _ in $(seq 180); do [ -f "$SPLIT_DIR/tasks.jsonl" ] && break; sleep 30; done
[ -f "$SPLIT_DIR/tasks.jsonl" ] || { echo "no split at $SPLIT_DIR"; exit 1; }

export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# JIT builds (FlashInfer in vLLM, kernels in the trainer) call `ninja` from PATH, as in W1.
export PATH=$W2_VENV/bin:$PATH
# harbor_env.harness is not in the OpenEnv wheel; termgrade_env.py is imported by the spawned rollout process too.
export PYTHONPATH=$OPENENV/envs:$BENCH/workloads/agentic
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
export VLLM_URL=http://localhost:$PORT HARBOR_URL=http://localhost:$((PORT + 2))
# Harbor shard 0 keeps PORT+2 (capture PORT+3); shard i >= 1 listens on HPORT+2i (capture +1), outside PORT's 10-port block.
HARBOR_SHARDS=${HARBOR_SHARDS:-1}; TERMINUS_MAX_TURNS=${TERMINUS_MAX_TURNS:-0}
HPORT=$((20000 + (SLURM_JOB_ID % 500) * 20))
HARBOR_PORTS=($((PORT + 2))); for i in $(seq 1 $((HARBOR_SHARDS - 1))); do HARBOR_PORTS+=($((HPORT + 2 * i))); done
[ "$HARBOR_SHARDS" -gt 1 ] && export HARBOR_URLS=$(printf 'http://localhost:%s,' "${HARBOR_PORTS[@]}" | sed 's/,$//')
VLLM_GPUS=$(seq -s, 0 $((NVLLM - 1)))
TRAIN_GPUS=$(seq -s, $NVLLM $((NVLLM + NTRAIN - 1)))
# Harbor writes one directory per trial (agent logs, verifier output, result.json): node-local, archived at the end.
export OPENENV_HARBOR_TRIALS_DIR=/scratch/$USER/w2-trials-$SLURM_JOB_ID
mkdir -p "$OPENENV_HARBOR_TRIALS_DIR"
[ -d /scratch ] && export VLLM_CACHE_ROOT=/scratch/$USER/vllm-cache-$SLURM_JOB_ID

# provenance
cp "$BENCH"/configs/trl/w2-"$SIZE".* "$0" "$BENCH/workloads/agentic/termgrade_env.py" "$RUN_DIR/" 2>/dev/null
cp "$SPLIT_DIR/tasks.jsonl" "$RUN_DIR/split_tasks.jsonl" 2>/dev/null
git -C /fsx/amine_dirhoussi/trl-wt-stable-bench rev-parse HEAD > "$RUN_DIR/trl_sha"
git -C /fsx/amine_dirhoussi/trl-wt-stable-bench diff HEAD -- trl > "$RUN_DIR/trl.diff"
uv pip freeze --python $W2_VENV/bin/python > "$RUN_DIR/freeze.txt"
env | grep -E '^(MAX_STALENESS|SANDBOX|W2_SANDBOX|STAGE|SIZE|PROBE_SIZES|EVAL_MODEL|TAG|NVLLM|NTRAIN|SPLIT|MAX_STEPS|SAVE_STEPS|CONCURRENCY|AGENT_TIMEOUT|STEP_LIMIT|MAX_MODEL_LEN|SAMPLES|N_TASKS|HARBOR_SHARDS|TERMINUS_MAX_TURNS|TOKEN_BUDGET|GRAD_ACCUM|PROMPTS_PER_STEP|NUM_GENERATIONS)=' > "$RUN_DIR/launch.env"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$RUN_DIR/nodes"

( echo "node,timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 \
    | sed -u "s/^/$(hostname),/" ) > "$RUN_DIR/gpu.csv" &
SMI_PID=$!

# Chat completions for terminus-2 (through the capture proxy): token ids in the reply, processed logprobs, thinking
# off (SHARED_W2), the trainer's sampling params only (--generation-config vllm). No images. Its own process group, so
# a probe over several models can stop it completely between them.
# FlashInfer's JIT cache (~/.cache/flashinfer/<version>/<arch>/cached_ops) is keyed by version only, but each module's
# build.ninja holds the venv's paths: bench-trl and bench-trl-ep (same FlashInfer 0.7.2.post1) rebuild each other's MoE
# kernel (fused_moe_trtllm_sm100, ~7 min) on every switch, and two starting at once kept invalidating each other until
# both hit the engine timeout (372822 / 372826, envs/trl/NOTES_MOE.md item 31). One workspace per vLLM venv,
# seeded once from the home cache.
FI_WS=/fsx/amine_dirhoussi/flashinfer-ws/$(basename $VLLM_VENV)
mkdir -p "$FI_WS/.cache"
flock "$FI_WS.lock" bash -c "[ -d '$FI_WS/.cache/flashinfer' ] || cp -a '$HOME/.cache/flashinfer' '$FI_WS/.cache/'"
start_vllm() {  # start_vllm <model> <log>
    PATH=$VLLM_VENV/bin:$PATH CUDA_VISIBLE_DEVICES=$VLLM_GPUS VLLM_SERVER_DEV_MODE=1 FLASHINFER_WORKSPACE_BASE=$FI_WS setsid vllm serve "$1" \
        --port $PORT --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 \
        --data-parallel-size $NVLLM --api-server-count $((NVLLM * 2)) --seed 42 \
        --logprobs-mode processed_logprobs --generation-config vllm --return-tokens-as-token-ids \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --limit-mm-per-prompt '{"image":0,"video":0}' \
        --weight-transfer-config '{"backend":"nccl"}' ${VLLM_EXTRA:-} > "$2" 2>&1 &
    VLLM_PID=$!
}
VLLM_PID=""
wait_vllm() {  # wait_vllm <log>: a failed engine can leave vLLM's main process hanging, so also watch the log
    until curl -sf "$VLLM_URL/health" > /dev/null; do
        if ! kill -0 $VLLM_PID 2>/dev/null || grep -q "Engine core initialization failed" "$1"; then
            echo "vLLM died during startup"; grep -m3 -E "Error|error" "$1"; exit 1
        fi
        sleep 5
    done
}
stop_vllm() { [ -n "$VLLM_PID" ] || return 0; kill -- -$VLLM_PID 2>/dev/null; sleep 10; kill -9 -- -$VLLM_PID 2>/dev/null; }

# The Harbor server: terminus-2 runs host-side here, so the capture proxy is reached on localhost (--expose direct, no
# tunnel). One env session per in-flight rollout, plus the factory's metadata connection, plus slack. The model and
# engine are passed per rollout, so one server serves every model a probe runs. It checks the engine at startup, so it
# starts once vLLM answers.
HARBOR_PID=""
HARBOR_ENV=()
if [ "$SANDBOX" = local ]; then
    # Unprivileged Apptainer for this node's OS (container-tools/apptainer-env.sh), on PATH as `singularity`. OpenEnv's
    # server passes no environment kwargs, so Harbor runs its default `--no-mount home,tmp,bind-paths`. On the GPU nodes
    # (Apptainer 1.4.5) DNS still works with it (job 365442); on Apptainer 1.5.5 it drops /etc/resolv.conf and the in-box
    # pip bootstrap fails (365392), so APPTAINER_BIND (Apptainer's env form of --bind) binds it explicitly either way.
    # Box scratch (Harbor's staging dirs) and Apptainer's own cache/tmp go under one per-job dir on /scratch.
    source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh > "$RUN_DIR/apptainer-env.log" 2>&1
    BOX_DIR=/scratch/$USER/w2-boxes-$SLURM_JOB_ID
    mkdir -p "$BOX_DIR/tmp" "$BOX_DIR/apptainer-cache" "$BOX_DIR/apptainer-tmp"
    HARBOR_ENV=(LD_LIBRARY_PATH=$AP_LIBS APPTAINER_BIND=/etc/resolv.conf TMPDIR=$BOX_DIR/tmp
                APPTAINER_CACHEDIR=$BOX_DIR/apptainer-cache APPTAINER_TMPDIR=$BOX_DIR/apptainer-tmp)
    apptainer --version > "$RUN_DIR/apptainer_version" 2>&1
fi
HARBOR_PIDS=()
start_harbor() {
    # TERMINUS_MAX_TURNS=1: the same CLI through harbor_serve.py, which gives terminus-2 the rollout's step limit.
    local cli=($W2_VENV/bin/openenv)
    [ "$TERMINUS_MAX_TURNS" = 1 ] && cli=($W2_VENV/bin/python "$BENCH/workloads/agentic/harbor_serve.py")
    for i in "${!HARBOR_PORTS[@]}"; do
        local log=$RUN_DIR/harbor.log; [ $i -gt 0 ] && log=$RUN_DIR/harbor-$i.log
        env "${HARBOR_ENV[@]}" MAX_CONCURRENT_ENVS=$((CONCURRENCY + 8)) "${cli[@]}" harbor serve \
            --dataset "$SPLIT_DIR" --llm-url "$VLLM_URL" --model "$1" \
            --host 127.0.0.1 --port ${HARBOR_PORTS[$i]} --capture-port $((HARBOR_PORTS[$i] + 1)) --expose direct \
            --max-output-tokens 8192 --no-rollouts --no-run-history > "$log" 2>&1 &
        HARBOR_PIDS+=($!)
    done
    HARBOR_PID=${HARBOR_PIDS[0]}
    for i in "${!HARBOR_PORTS[@]}"; do
        until curl -sf "http://localhost:${HARBOR_PORTS[$i]}/health" > /dev/null; do
            kill -0 ${HARBOR_PIDS[$i]} 2>/dev/null || { echo "harbor server $i died during startup"; tail -n 30 "$RUN_DIR"/harbor*.log; exit 1; }
            sleep 5
        done
        # local boxes: their CPU and memory, sampled every 5 s (boxes.csv for shard 0, boxes-<i>.csv for the others)
        local csv=$RUN_DIR/boxes.csv; [ $i -gt 0 ] && csv=$RUN_DIR/boxes-$i.csv
        [ "$SANDBOX" = local ] && $W2_VENV/bin/python "$BENCH/workloads/agentic/box_usage.py" ${HARBOR_PIDS[$i]} 5 > "$csv" &
    done
    echo "harbor_ready_s=$(echo "$(date +%s.%N) - $JOB_START" | bc)" | tee "$RUN_DIR/harbor_ready"
    echo "harbor_shards=$HARBOR_SHARDS terminus_max_turns=$TERMINUS_MAX_TURNS" | tee -a "$RUN_DIR/harbor_ready"
    # Each server's CPU (pid, %cpu over its life, cumulative cpu time), every 60 s: the event-loop load per shard.
    ( while sleep 60; do echo "$(date +%s) $(ps -o pid=,pcpu=,cputime= -p "$(IFS=,; echo "${HARBOR_PIDS[*]}")" | tr '\n' ' ')"; done ) > "$RUN_DIR/harbor_cpu.txt" &
}

# Every process under the Harbor server (local boxes are its `singularity exec` children and their trees).
tree_pids() { ps -u $USER -o pid=,ppid= | awk -v r=$1 '{p[$1]=$2} END {do {c=0; for (x in p) if (p[x] in s || p[x]==r) if (!(x in s)) {s[x]=1; c++}} while (c); for (x in s) print x}'; }
cleanup() {
    BOX_PIDS=""
    if [ "$SANDBOX" = local ]; then for p in "${HARBOR_PIDS[@]}"; do BOX_PIDS="$BOX_PIDS $(tree_pids $p)"; done; fi
    stop_vllm; kill "${HARBOR_PIDS[@]}" $SMI_PID 2>/dev/null; pkill -P $$ 2>/dev/null
    sleep 5
    if [ "$SANDBOX" = local ]; then
        # Local boxes are processes of this job: stop any the server left, then remove their scratch and caches.
        kill ${BOX_PIDS:-} 2>/dev/null; pkill -u $USER -f "$BOX_DIR" 2>/dev/null; sleep 5
        kill -9 ${BOX_PIDS:-} 2>/dev/null; pkill -9 -u $USER -f "$BOX_DIR" 2>/dev/null
        echo "box processes at cleanup: $(echo ${BOX_PIDS:-} | wc -w); left after: $(pgrep -u $USER -fc "$BOX_DIR")" > "$RUN_DIR/sweep.log"
        rm -rf "$BOX_DIR"
    else
        # Sandboxes outlive a killed server: sweep (twice) every run, whatever its exit.
        "$BENCH/workloads/agentic/sweep.sh" $W2_VENV/bin/python "$RUN_DIR/sweep.log"
    fi
    tar czf "$RUN_DIR/trials.tgz" -C "$OPENENV_HARBOR_TRIALS_DIR" --exclude='*/agent/*.cast' . 2>/dev/null && rm -rf "$OPENENV_HARBOR_TRIALS_DIR"
    date +%s.%N > "$RUN_DIR/job_end"
}
trap cleanup EXIT

cd "$BENCH"
if [ "$STAGE" = probe ]; then
    # PROBE_SIZES: the models to probe one after the other on the same tasks (default: SIZE). EVAL_MODEL=<checkpoint dir>
    # evaluates that checkpoint instead (W2 eval: SPLIT=eval), recorded as probe-ckpt.jsonl.
    [ -n "${EVAL_MODEL:-}" ] && { MODELS[ckpt]=$EVAL_MODEL; PROBE_SIZES=ckpt; }
    for size in ${PROBE_SIZES:-$SIZE}; do
        start_vllm "${MODELS[$size]}" "$RUN_DIR/vllm-$size.log"
        wait_vllm "$RUN_DIR/vllm-$size.log"
        echo "vllm_ready_s[$size]=$(echo "$(date +%s.%N) - $JOB_START" | bc)" | tee -a "$RUN_DIR/vllm_ready"
        [ -n "$HARBOR_PID" ] || start_harbor "${MODELS[$size]}"
        $W2_VENV/bin/python workloads/agentic/probe.py --server "$HARBOR_URL" --vllm-url "$VLLM_URL" --model "${MODELS[$size]}" \
            --split "$SPLIT_DIR" --samples ${SAMPLES:-4} --n-tasks ${N_TASKS:-0} --concurrency $CONCURRENCY \
            --agent-timeout $AGENT_TIMEOUT --agent-step-limit $STEP_LIMIT --out "$RUN_DIR/probe-$size.jsonl" 2>&1 | tee "$RUN_DIR/probe-$size.log"
        rc=${PIPESTATUS[0]}
        curl -s "$VLLM_URL/metrics" | grep -E "^vllm:(num_preemptions_total|prefix_cache_(queries|hits)_total)" > "$RUN_DIR/vllm_final_metrics-$size.txt"
        stop_vllm
    done
else
    start_vllm "$MODEL" "$RUN_DIR/vllm.log"
    wait_vllm "$RUN_DIR/vllm.log"
    echo "vllm_ready_s=$(echo "$(date +%s.%N) - $JOB_START" | bc)" | tee "$RUN_DIR/vllm_ready"
    start_harbor "$MODEL"
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS $W2_VENV/bin/accelerate launch --config_file "configs/trl/w1-S.accelerate.yaml" \
        --num_processes $NTRAIN --main_process_port $((PORT + 7)) "configs/trl/w2-$SIZE.py" 2>&1 | tee "$RUN_DIR/train.log"
    rc=${PIPESTATUS[0]}
    curl -s "$VLLM_URL/metrics" | grep -E "^vllm:(num_preemptions_total|prefix_cache_(queries|hits)_total|cache_config_info)" > "$RUN_DIR/vllm_final_metrics.txt"
fi
echo "exit code $rc"
exit $rc
