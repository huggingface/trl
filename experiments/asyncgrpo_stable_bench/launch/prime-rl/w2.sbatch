#!/bin/bash
# W2 agentic run for prime-rl (TermGrade, terminus-2, Qwen3.5-9B): 1 node, 4 inference + 4 trainer GPUs, 60 steps,
# HF exports at 20/40/60. Sandboxes are verifiers' `docker` runtime against rootless podman on this node.
# Usage: b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/prime-rl/w2.sbatch [arm] [extra rl args]
#   run_id = prime-rl-w2-M-<arm>-<job id>; arm defaults to `shared` (configs/prime-rl/w2-M.toml), a suffixed arm
#   (`shared-smoke`) reuses that config under its own run_id.
# Env knobs: TAG (appended to the run_id), MAX_STEPS (-> --max-steps), SAVE_STEPS (-> --ckpt.interval), MAX_INFLIGHT
# (-> concurrency cap, e.g. 16 for a smoke; batch_size stays 64), EXPORT=0 skips the HF export, SIZE (default M; `moe` ->
# w2-moe.toml, Qwen3.5-35B-A3B + LoRA), WEIGHT_BROADCAST (default nccl; `auto` passes no --weight-broadcast.* so prime-rl
# picks filesystem for LoRA), KEEP_ADAPTERS (e.g. "20 40 60": copy the LoRA adapter prime-rl broadcasts after those steps
# out of its rolling broadcasts/ dir, then merge each into a full HF dir hf/step_N with envs/prime-rl/merge_lora.py),
# MERGE_CHECK=1 (load the last merged dir in vLLM and answer one prompt).
# Environment tooling only (no library change): podman service + docker CLI shim (container-tools/podman-env.sh), the
# TermGrade base image loaded from a tar, ~/.cache/harbor/<dataset> symlinks to the local task copies, and the env
# worker pool started under `unshare -r` through a PATH wrapper (verifiers' stage_tests tar keeps the host uid, which
# rootless podman can't map; NATIVE_SANDBOXES.md, finding 1).
#SBATCH --job-name=prl-w2
#SBATCH --partition=hopper-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out

set -uo pipefail
ARM=${1:-${ARM:-shared}}; shift || true
SIZE=${SIZE:-M}
SRC=/fsx/amine_dirhoussi/stable-bench-src/prime-rl
VENV=/fsx/amine_dirhoussi/envs/bench-prime-rl
TOOLS=/fsx/amine_dirhoussi/stable-bench-src/container-tools
LOCAL=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local
export BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
CFG=$BENCH/configs/prime-rl/w2-$SIZE.toml
[ "${ARM%%-*}" = shared ] || CFG=$BENCH/configs/prime-rl/w2-$SIZE-${ARM%%-*}.toml
RUNS=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl
RUN_ID=prime-rl-w2-$SIZE-$ARM${TAG:+-$TAG}-$SLURM_JOB_ID
OUT=$RUNS/$RUN_ID
AUX=$OUT-aux # sidecar files: rl refuses to start in a non-empty run dir
mkdir -p "$AUX/bin"
EXTRA=()
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")
[ -n "${SAVE_STEPS:-}" ] && EXTRA+=(--ckpt.interval "$SAVE_STEPS")
WB_ARGS=()
[ "${WEIGHT_BROADCAST:-nccl}" = nccl ] && WB_ARGS=(--weight-broadcast.type nccl --weight-broadcast.port $((8000 + (SLURM_JOB_ID % 90) * 10 + 5)))
[ -n "${MAX_INFLIGHT:-}" ] && EXTRA+=(--orchestrator.concurrency.initial-inflight "$MAX_INFLIGHT" --orchestrator.concurrency.max-inflight "$MAX_INFLIGHT")

source "$VENV/bin/activate"
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))

# Rootless podman with a Docker-compatible socket under /scratch; on podman 3.x this also puts the `docker run --rm`
# shim first on PATH. Every container, volume and image this job creates is removed at exit.
source "$TOOLS/podman-env.sh"
cleanup() {
    trap - EXIT TERM INT
    kill $SMI $SAMPLER $KEEPER 2>/dev/null
    echo "cleanup $(date +%s.%N) containers=$(docker ps -aq 2>/dev/null | wc -l)" | tee -a "$AUX/job_times.txt"
    for _ in 1 2 3; do docker ps -aq 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1; done
    kill "$PODMAN_SERVICE_PID" 2>/dev/null; wait "$PODMAN_SERVICE_PID" 2>/dev/null
    # Not `podman system reset`: it would also drop the per-user CNI configs in ~/.config/cni, shared with my podman jobs on other nodes.
    podman rm -af >/dev/null 2>&1; podman volume prune -f >/dev/null 2>&1; podman rmi -af >/dev/null 2>&1
    rm -rf "$XDG_RUNTIME_DIR"
    echo "cleanup_done $(date +%s.%N) containers_left=$(podman ps -aq 2>/dev/null | wc -l)" | tee -a "$AUX/job_times.txt"
}
SMI=; SAMPLER=; KEEPER=; PODMAN_SERVICE_PID=
trap cleanup EXIT
trap 'exit 143' TERM INT
podman_service_start "$AUX/podman-service.log" || exit 1
echo "podman $(podman --version | awk '{print $3}') graphroot=$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null) engine=$(docker version --format '{{.Server.Version}}')"
podman load -q -i "$TOOLS/images/termgrade-base.tar"

# HarborTaskset reads ~/.cache/harbor/<dataset> as an already-downloaded dataset (no registry needed).
mkdir -p ~/.cache/harbor
ln -sfn "$LOCAL/train/tasks" ~/.cache/harbor/termgrade-train
ln -sfn "$LOCAL/eval/tasks" ~/.cache/harbor/termgrade-eval
ln -sfn "$LOCAL/hard-train/tasks" ~/.cache/harbor/termgrade-hard-train  # SIZE=moehard (workloads/agentic/hard/README.md)
ln -sfn "$LOCAL/hard-eval/tasks" ~/.cache/harbor/termgrade-hard-eval

# rl starts each env worker pool as `env-worker-pool @ <cfg>` from PATH; this wrapper runs it in a user namespace
# (host uid -> 0) so the test tarballs verifiers writes carry uid 0. Nothing else runs in the namespace.
cat > "$AUX/bin/env-worker-pool" <<EOF
#!/bin/bash
exec unshare -r "$VENV/bin/env-worker-pool" "\$@"
EOF
chmod +x "$AUX/bin/env-worker-pool"
export PATH=$AUX/bin:$PATH

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 \
    | sed -u "1s/\$/, node/; 2,\$s/\$/, $(hostname)/" > "$AUX/gpu.csv" &
SMI=$!
# host load: unix time, used MB, live containers, 1-min load average, every 5 s
( while true; do echo "$(date +%s),$(free -m | awk '/Mem/{print $3}'),$(podman ps -q 2>/dev/null | wc -l),$(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > "$AUX/host.csv" &
SAMPLER=$!

# prime-rl keeps only the last two broadcasts/step_N; copy the wanted adapters once their .finished marker is written.
if [ -n "${KEEP_ADAPTERS:-}" ]; then
    ( todo="$KEEP_ADAPTERS"; while [ -n "$todo" ]; do left=""; for n in $todo; do
        d=$OUT/broadcasts/step_$n
        if [ -f "$d/.finished" ]; then cp -r "$d" "$AUX/adapter_step_$n" && echo "adapter_kept $(date +%s.%N) step=$n" >> "$AUX/job_times.txt"
        else left="$left $n"; fi; done; todo=${left# }; sleep 10; done ) &
    KEEPER=$!
fi

echo "job_start $(date +%s.%N) run_id=$RUN_ID node=$(hostname) port=$PORT cfg=$CFG" | tee "$AUX/job_times.txt"
cd "$SRC"
rl @ "$CFG" \
    --output-dir "$RUNS" \
    --run.name "$RUN_ID" \
    --inference.server.port "$PORT" \
    --orchestrator.model.client.base-url "http://localhost:$PORT/v1" \
    --rollout-transport.type zmq --rollout-transport.port $((PORT + 3)) \
    "${WB_ARGS[@]}" \
    --monitors.wandb.project stable-bench-prime-rl --monitors.wandb.name "$RUN_ID" \
    --no-monitors.prime \
    "${EXTRA[@]}" "$@"
rc=$?
echo "train_end $(date +%s.%N) rc=$rc" | tee -a "$AUX/job_times.txt"

# Stop the sandboxes before the export (it needs no containers), then export each DCP checkpoint to a complete HF dir
# (bench copy of prime-rl's converter + complete_hf.py, as w1.sbatch).
for _ in 1 2 3; do docker ps -aq 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1; done
if [ "${EXPORT:-1}" = 1 ]; then
    for step_dir in "$OUT"/checkpoints/step_*; do
        [ -d "$step_dir/trainer" ] || continue
        step=${step_dir##*_}
        echo "export_start $(date +%s.%N) step=$step" | tee -a "$AUX/job_times.txt"
        torchrun --nproc-per-node 8 --master-port $((PORT + 7)) "$BENCH/envs/prime-rl/convert_dcp_to_bf16.py" "$step_dir" "$OUT/hf/step_$step" \
            && python "$BENCH/envs/prime-rl/complete_hf.py" "$OUT/hf/step_$step" \
            || echo "export of step $step failed"
        echo "export_end $(date +%s.%N) step=$step" | tee -a "$AUX/job_times.txt"
    done
    for a in "$AUX"/adapter_step_*; do
        [ -d "$a" ] || continue
        step=${a##*_}
        echo "export_start $(date +%s.%N) step=$step" | tee -a "$AUX/job_times.txt"
        python "$BENCH/envs/prime-rl/merge_lora.py" "$a" "$OUT/hf/step_$step" || echo "merge of step $step failed"
        echo "export_end $(date +%s.%N) step=$step" | tee -a "$AUX/job_times.txt"
    done
    # MERGE_CHECK=1 (smokes): serve the last merged dir in vLLM on 1 GPU and answer one prompt.
    [ "${MERGE_CHECK:-0}" = 1 ] && [ -n "${step:-}" ] && [ -f "$OUT/hf/step_$step/config.json" ] \
        && CUDA_VISIBLE_DEVICES=0 python "$BENCH/envs/prime-rl/vllm_load_check.py" "$OUT/hf/step_$step" 2>&1 | grep -E "vLLM architecture|greedy|boxed|Error"
fi
echo "job_end $(date +%s.%N)" | tee -a "$AUX/job_times.txt"
exit $rc
