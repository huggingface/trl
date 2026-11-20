#!/bin/bash
# W1 math run for prime-rl: 1 node, 4 inference + 4 trainer GPUs, 200 steps, HF exports at 50/100/150/200.
# Usage: b200-queue launch/prime-rl/w1.sbatch [S|M] [shared|recommended] [extra rl args]   (run_id = prime-rl-w1-<size>-<arm>-<job id>)
# shared -> configs/prime-rl/w1-<size>.toml, recommended -> configs/prime-rl/w1-<size>-recommended.toml
# Env knobs (positional args win): SIZE (S|M, default S), ARM (default shared), TAG (appended to the run_id),
# MAX_STEPS (-> --max-steps, for smokes), SAVE_STEPS (-> --ckpt.interval, e.g. 3 to time a checkpoint save in a smoke).
#SBATCH --job-name=prl-w1
#SBATCH --partition=hopper-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=14:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out

set -uo pipefail
SIZE=${1:-${SIZE:-S}}; ARM=${2:-${ARM:-shared}}; shift 2 || shift || true
SMOKE_ARGS=()
[ -n "${MAX_STEPS:-}" ] && SMOKE_ARGS+=(--max-steps "$MAX_STEPS")
[ -n "${SAVE_STEPS:-}" ] && SMOKE_ARGS+=(--ckpt.interval "$SAVE_STEPS")
SRC=/fsx/amine_dirhoussi/stable-bench-src/prime-rl
VENV=/fsx/amine_dirhoussi/envs/bench-prime-rl
export BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
CFG=$BENCH/configs/prime-rl/w1-$SIZE.toml
# The config comes from the arm before any "-" suffix, so `shared-smoke` runs w1-<size>.toml under its own run_id.
[ "${ARM%%-*}" = shared ] || CFG=$BENCH/configs/prime-rl/w1-$SIZE-${ARM%%-*}.toml
RUNS=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl
RUN_ID=prime-rl-w1-$SIZE-$ARM${TAG:+-$TAG}-$SLURM_JOB_ID
OUT=$RUNS/$RUN_ID
AUX=$OUT-aux # sidecar files (gpu.csv, job times): rl refuses to start in a non-empty run dir
mkdir -p "$AUX"

source "$VENV/bin/activate"
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONPATH=$BENCH/configs/prime-rl${PYTHONPATH:+:$PYTHONPATH}
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 \
    | sed -u "1s/\$/, node/; 2,\$s/\$/, $(hostname)/" > "$AUX/gpu.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

echo "job_start $(date +%s.%N) run_id=$RUN_ID node=$(hostname) port=$PORT cfg=$CFG" | tee "$AUX/job_times.txt"
cd "$SRC"
rl @ "$CFG" \
    --output-dir "$RUNS" \
    --run.name "$RUN_ID" \
    --inference.server.port "$PORT" \
    --orchestrator.model.client.base-url "http://localhost:$PORT/v1" \
    --rollout-transport.type zmq --rollout-transport.port $((PORT + 3)) \
    --weight-broadcast.type nccl --weight-broadcast.port $((PORT + 5)) \
    --monitors.wandb.project stable-bench-prime-rl --monitors.wandb.name "$RUN_ID" \
    --no-monitors.prime \
    "${SMOKE_ARGS[@]}" "$@"
rc=$?
echo "train_end $(date +%s.%N) rc=$rc" | tee -a "$AUX/job_times.txt"

# The trainer writes DCP checkpoints only; export each to a complete HF bf16 dir for workloads/math/eval.py, with the
# bench copy of prime-rl's converter (upstream raises on Qwen3.5's tied lm_head, NOTES.md #19), then fill in any base
# tensor the export lacks (SHARED_W1 rev. 6).
for step_dir in "$OUT"/checkpoints/step_*; do
    [ -d "$step_dir/trainer" ] || continue
    step=${step_dir##*_}
    echo "export_start $(date +%s.%N) step=$step" | tee -a "$AUX/job_times.txt"
    torchrun --nproc-per-node 8 --master-port $((PORT + 7)) "$BENCH/envs/prime-rl/convert_dcp_to_bf16.py" "$step_dir" "$OUT/hf/step_$step" \
        && python "$BENCH/envs/prime-rl/complete_hf.py" "$OUT/hf/step_$step" \
        || echo "export of step $step failed"
    echo "export_end $(date +%s.%N) step=$step" | tee -a "$AUX/job_times.txt"
done
echo "job_end $(date +%s.%N)" | tee -a "$AUX/job_times.txt"
exit $rc
