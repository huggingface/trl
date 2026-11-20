#!/bin/bash
# W2 agentic (TermGrade, terminus-2 through OpenEnv Harbor, local Apptainer boxes) on a MoE model with expert
# parallelism + FSDP (sharded at load time by transformers), LoRA, one node: vLLM (native impl, DP=NVLLM + EP) on the
# first NVLLM GPUs, the trainer (torchrun, DistributedConfig(fsdp_size=NTRAIN, ep_size=EP)) on the next NTRAIN.
# launch/trl/w2.sbatch (SANDBOX=local) for the W2 side, launch/trl/moe_w1.sbatch for the MoE side.
# No separate smoke (user): launch/trl/w2_watchdog.py checks the first WD_STEPS steps, the first episodes and the first checkpoint
# merge inside the job and stops the trainer with a clear message (watchdog.txt, WATCHDOG_FAILED) on failure.
#   run:   PRIO_TAG=PRIO_TRAIN b200-queue --gres=gpu:8 --exclusive -t 1-00:00:00 $PWD/moe_w2.sbatch
# Env: MODEL (Qwen/Qwen3.5-35B-A3B), DTYPE (bfloat16), LORA_R (32), LR, EP (= NTRAIN), NVLLM (4), NTRAIN (4), TAG (local),
# MAX_STEPS (60), SAVE_STEPS (20), MAX_STALENESS (4), GRAD_ACCUM (14), TOKEN_BUDGET (65536), CONCURRENCY (64),
# HARBOR_SHARDS (4: Harbor servers, rollouts spread round-robin; 1 = the single server of the W2 runs), TERMINUS_MAX_TURNS
# (1: terminus-2 stops at STEP_LIMIT calls itself, workloads/agentic/harbor_serve.py; 0 = only the capture proxy's cap),
# AGENT_TIMEOUT (600), STEP_LIMIT (30), MAX_MODEL_LEN (65536), SPLIT (train),
# WD_STEPS (3), WD_FIRST_STEP_S (5400): watchdog checks and the deadline for step 1 after the trainer starts.
# Every checkpoint is merged into a full HF model (merge_lora.py) in stable-bench-runs/trl-moe-ep/merged/ for the W2 eval.
# Job name must not start with trl-w2 (workloads/agentic/sweep.sh, envs/trl/NOTES.md "W2 local" item 4).
#SBATCH -J w2local-trl-moe
#SBATCH --nodes=1
#SBATCH -t 24:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/jobs/%x-%j.out
set -uo pipefail
JOB_START=$(date +%s.%N)
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/trl
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
VLLM_VENV=/fsx/amine_dirhoussi/envs/bench-trl-ep    # vLLM 0.33, the MoE runs' server stack
W2_VENV=/fsx/amine_dirhoussi/envs/bench-trl-w2      # trainer + OpenEnv Harbor server (mcp<2); same torch/transformers/peft pins
TRL=/fsx/amine_dirhoussi/trl-wt-asyncgrpo-ep        # on PYTHONPATH, ahead of the venv's editable stable-bench TRL
OPENENV=/fsx/amine_dirhoussi/stable-bench-src/openenv
export MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B} DTYPE=${DTYPE:-bfloat16} LORA_R=${LORA_R:-32}
NVLLM=${NVLLM:-4}; NTRAIN=${NTRAIN:-4}; export EP=${EP:-$NTRAIN} TAG=${TAG:-local}
export MAX_STEPS=${MAX_STEPS:-60} SAVE_STEPS=${SAVE_STEPS:-20} MAX_STALENESS=${MAX_STALENESS:-4}
export GRAD_ACCUM=${GRAD_ACCUM:-14} TOKEN_BUDGET=${TOKEN_BUDGET:-65536}
export CONCURRENCY=${CONCURRENCY:-64} AGENT_TIMEOUT=${AGENT_TIMEOUT:-600} STEP_LIMIT=${STEP_LIMIT:-30}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536} SPLIT=${SPLIT:-train} HARBOR_SHARDS=${HARBOR_SHARDS:-4} TERMINUS_MAX_TURNS=${TERMINUS_MAX_TURNS:-1}
export RUN_ID=trl-moe-w2-$(basename "$MODEL")-ep$EP-$DTYPE-lora$LORA_R-$TAG-$SLURM_JOB_ID
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/$RUN_ID
mkdir -p "$RUN_DIR"
echo "$JOB_START" > "$RUN_DIR/job_start"
# Local boxes: the tasks' docker_image is the .sif of the TermGrade base image (envs/trl/NOTES.md "W2 local").
export SPLIT_DIR=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/$SPLIT W2_SANDBOX=singularity
[ -f "$SPLIT_DIR/tasks.jsonl" ] || { echo "no split at $SPLIT_DIR"; exit 1; }

export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# A cold start JIT-builds FlashInfer's MoE kernel (~6-7 min, envs/trl/NOTES_MOE.md items 10 and 31).
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export PATH=$W2_VENV/bin:$PATH
export PYTHONPATH=$TRL:$OPENENV/envs:$BENCH/workloads/agentic
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
export VLLM_URL=http://localhost:$PORT
# Harbor shard i listens on HPORT + 2i, its capture proxy on HPORT + 2i + 1 (outside the 10-port block of PORT).
HPORT=$((20000 + (SLURM_JOB_ID % 500) * 20))
URLS=(); for i in $(seq 0 $((HARBOR_SHARDS - 1))); do URLS+=(http://127.0.0.1:$((HPORT + 2 * i))); done
export HARBOR_URL=${URLS[0]} HARBOR_URLS=$(IFS=,; echo "${URLS[*]}")
export OPENENV_HARBOR_TRIALS_DIR=/scratch/$USER/w2-trials-$SLURM_JOB_ID
mkdir -p "$OPENENV_HARBOR_TRIALS_DIR"
export VLLM_CACHE_ROOT=/scratch/$USER/vllm-cache-$SLURM_JOB_ID

# provenance
cp "$HERE/moe_w2.py" "$0" "$BENCH/workloads/agentic/termgrade_env.py" "$RUN_DIR/" 2>/dev/null
cp "$SPLIT_DIR/tasks.jsonl" "$RUN_DIR/split_tasks.jsonl"
git -C $TRL rev-parse HEAD > "$RUN_DIR/trl_sha"; git -C $TRL diff HEAD > "$RUN_DIR/trl.diff"
uv pip freeze --python $W2_VENV/bin/python > "$RUN_DIR/freeze.txt"
env | grep -E '^(MODEL|DTYPE|LORA_R|LR|EP|TAG|MAX_STEPS|SAVE_STEPS|MAX_STALENESS|GRAD_ACCUM|TOKEN_BUDGET|CONCURRENCY|AGENT_TIMEOUT|STEP_LIMIT|MAX_MODEL_LEN|SPLIT|HARBOR_SHARDS|TERMINUS_MAX_TURNS|WD_STEPS|WD_FIRST_STEP_S)=' > "$RUN_DIR/launch.env"
hostname > "$RUN_DIR/nodes"

( echo "node,timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 \
    | sed -u "s/^/$(hostname),/" ) > "$RUN_DIR/gpu.csv" &
SMI_PID=$!

# FlashInfer's JIT cache (~/.cache/flashinfer/<version>/<arch>/cached_ops) is keyed by version only, but each module's
# build.ninja holds the venv's paths: bench-trl and bench-trl-ep (same FlashInfer 0.7.2.post1) rebuild each other's MoE
# kernel (fused_moe_trtllm_sm100, ~7 min) on every switch, and two starting at once kept invalidating each other until
# both hit the engine timeout (372822 / 372826, envs/trl/NOTES_MOE.md item 31). One workspace per vLLM venv,
# seeded once from the home cache.
FI_WS=/fsx/amine_dirhoussi/flashinfer-ws/$(basename $VLLM_VENV)
mkdir -p "$FI_WS/.cache"
flock "$FI_WS.lock" bash -c "[ -d '$FI_WS/.cache/flashinfer' ] || cp -a '$HOME/.cache/flashinfer' '$FI_WS/.cache/'"
# vLLM: launch/trl/moe_w1.sbatch's MoE server (native impl, DP + EP, NCCL weight transfer) with launch/trl/w2.sbatch's W2
# flags (token ids in chat replies for the capture proxy, thinking off, 64k context).
PATH=$VLLM_VENV/bin:$PATH CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NVLLM - 1))) HF_HUB_OFFLINE=1 VLLM_SERVER_DEV_MODE=1 \
    FLASHINFER_WORKSPACE_BASE=$FI_WS setsid vllm serve "$MODEL" \
    --port $PORT --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.85 \
    --data-parallel-size $NVLLM --enable-expert-parallel --api-server-count $((NVLLM * 2)) --seed 42 \
    --logprobs-mode processed_logprobs --generation-config vllm --return-tokens-as-token-ids \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --weight-transfer-config '{"backend":"nccl"}' ${VLLM_EXTRA:-} > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
stop_vllm() { [ -n "$VLLM_PID" ] || return 0; kill -- -$VLLM_PID 2>/dev/null; sleep 10; kill -9 -- -$VLLM_PID 2>/dev/null; VLLM_PID=""; }

# Harbor servers on local Apptainer boxes, as launch/trl/w2.sbatch SANDBOX=local.
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh > "$RUN_DIR/apptainer-env.log" 2>&1
BOX_DIR=/scratch/$USER/w2-boxes-$SLURM_JOB_ID
mkdir -p "$BOX_DIR/tmp" "$BOX_DIR/apptainer-cache" "$BOX_DIR/apptainer-tmp"
HARBOR_ENV=(LD_LIBRARY_PATH=$AP_LIBS APPTAINER_BIND=/etc/resolv.conf TMPDIR=$BOX_DIR/tmp
            APPTAINER_CACHEDIR=$BOX_DIR/apptainer-cache APPTAINER_TMPDIR=$BOX_DIR/apptainer-tmp)
apptainer --version > "$RUN_DIR/apptainer_version" 2>&1
HARBOR_PIDS=()
HARBOR_CLI=($W2_VENV/bin/openenv)
[ "$TERMINUS_MAX_TURNS" = 1 ] && HARBOR_CLI=($W2_VENV/bin/python "$BENCH/workloads/agentic/harbor_serve.py")
tree_pids() { ps -u $USER -o pid=,ppid= | awk -v r=$1 '{p[$1]=$2} END {do {c=0; for (x in p) if (p[x] in s || p[x]==r) if (!(x in s)) {s[x]=1; c++}} while (c); for (x in s) print x}'; }
cleanup() {
    BOX_PIDS=""; for p in "${HARBOR_PIDS[@]}"; do BOX_PIDS="$BOX_PIDS $(tree_pids $p)"; done
    stop_vllm; kill "${HARBOR_PIDS[@]}" $SMI_PID 2>/dev/null; pkill -P $$ 2>/dev/null
    sleep 5
    kill $BOX_PIDS 2>/dev/null; pkill -u $USER -f "$BOX_DIR" 2>/dev/null; sleep 5
    kill -9 $BOX_PIDS 2>/dev/null; pkill -9 -u $USER -f "$BOX_DIR" 2>/dev/null
    echo "box processes at cleanup: $(echo $BOX_PIDS | wc -w); left after: $(pgrep -u $USER -fc "$BOX_DIR")" > "$RUN_DIR/sweep.log"
    rm -rf "$BOX_DIR"
    tar czf "$RUN_DIR/trials.tgz" -C "$OPENENV_HARBOR_TRIALS_DIR" --exclude='*/agent/*.cast' . 2>/dev/null && rm -rf "$OPENENV_HARBOR_TRIALS_DIR"
    rm -rf "$VLLM_CACHE_ROOT"
    date +%s.%N > "$RUN_DIR/job_end"
}
trap cleanup EXIT

until curl -sf "$VLLM_URL/health" > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null || grep -q "Engine core initialization failed" "$RUN_DIR/vllm.log"; then
        echo "vLLM died during startup"; tail -n 60 "$RUN_DIR/vllm.log"; exit 1
    fi
    sleep 5
done
echo "vllm_ready_s=$(echo "$(date +%s.%N) - $JOB_START" | bc)" | tee "$RUN_DIR/vllm_ready"

# One server per shard, each allowed every in-flight episode (round-robin keeps them near CONCURRENCY / shards).
# They check the engine at startup, so they start once vLLM answers.
for i in $(seq 0 $((HARBOR_SHARDS - 1))); do
    env "${HARBOR_ENV[@]}" MAX_CONCURRENT_ENVS=$((CONCURRENCY + 8)) "${HARBOR_CLI[@]}" harbor serve \
        --dataset "$SPLIT_DIR" --llm-url "$VLLM_URL" --model "$MODEL" \
        --host 127.0.0.1 --port $((HPORT + 2 * i)) --capture-port $((HPORT + 2 * i + 1)) --expose direct \
        --max-output-tokens 8192 --no-rollouts --no-run-history > "$RUN_DIR/harbor-$i.log" 2>&1 &
    HARBOR_PIDS+=($!)
done
for i in "${!URLS[@]}"; do
    until curl -sf "${URLS[$i]}/health" > /dev/null; do
        kill -0 ${HARBOR_PIDS[$i]} 2>/dev/null || { echo "harbor server $i died during startup"; tail -n 30 "$RUN_DIR/harbor-$i.log"; exit 1; }
        sleep 5
    done
    $W2_VENV/bin/python "$BENCH/workloads/agentic/box_usage.py" ${HARBOR_PIDS[$i]} 5 > "$RUN_DIR/boxes-$i.csv" &
done
echo "harbor_ready_s=$(echo "$(date +%s.%N) - $JOB_START" | bc) shards=$HARBOR_SHARDS terminus_max_turns=$TERMINUS_MAX_TURNS" | tee "$RUN_DIR/harbor_ready"
# Harbor server main-thread CPU, every 60 s: the single-loop bottleneck of envs/trl/W2_STEP_TIME.md, per shard.
( while sleep 60; do echo "$(date +%s) $(ps -o pid=,pcpu=,cputime= -p "$(IFS=,; echo "${HARBOR_PIDS[*]}")" | tr '\n' ' ')"; done ) > "$RUN_DIR/harbor_cpu.txt" &

cd "$HERE"
$W2_VENV/bin/python "$HERE/w2_watchdog.py" "$RUN_DIR" ${WD_STEPS:-3} ${WD_FIRST_STEP_S:-5400} > "$RUN_DIR/watchdog.log" 2>&1 &
WD_PID=$!
CUDA_VISIBLE_DEVICES=$(seq -s, $NVLLM $((NVLLM + NTRAIN - 1))) $W2_VENV/bin/torchrun --nproc_per_node $NTRAIN \
    --master_port $((PORT + 7)) moe_w2.py 2>&1 | tee "$RUN_DIR/train.log"
rc=${PIPESTATUS[0]}
echo "trainer exit code $rc"
curl -s "$VLLM_URL/metrics" | grep -E "^vllm:(num_preemptions_total|prefix_cache_(queries|hits)_total|cache_config_info)" > "$RUN_DIR/vllm_final_metrics.txt"

pkill -P $WD_PID 2>/dev/null; kill $WD_PID 2>/dev/null  # and a merge it may be running
[ -f "$RUN_DIR/WATCHDOG_FAILED" ] && echo "WATCHDOG FAILED: $(cat "$RUN_DIR/WATCHDOG_FAILED")"
# Checkpoints the watchdog did not merge yet (the last one: it merges a checkpoint only after the next step logs).
stop_vllm
for ckpt in "$RUN_DIR"/checkpoint-*; do
    [ -d "$ckpt" ] || continue
    out=/fsx/amine_dirhoussi/stable-bench-runs/trl-moe-ep/merged/$RUN_ID-$(basename "$ckpt")
    [ -f "$out/merge_check.json" ] && continue
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= $W2_VENV/bin/python "$HERE/merge_lora.py" "$ckpt" "$out" "$MODEL" > "$RUN_DIR/merge-${ckpt##*-}.log" 2>&1
    echo "merge $(basename "$ckpt"): exit $? -> $out"
done
exit $rc
