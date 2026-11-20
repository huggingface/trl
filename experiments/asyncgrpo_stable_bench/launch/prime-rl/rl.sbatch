#!/bin/bash
# Generic prime-rl `rl` launcher for smoke tests and recipe reproductions.
# Usage (through b200-queue): b200-queue --gres=gpu:2 --time=01:00:00 -J prl-smoke launch/prime-rl/rl.sbatch <config.toml> <tag> [extra rl args]
# Runs `rl @ <config.toml>` from the prime-rl checkout with the shared venv, a job-derived port,
# outputs under $RUNS/<run_id>/ and a 1 Hz gpu.csv next to them.
#SBATCH --job-name=prl-rl
#SBATCH --partition=hopper-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out

set -uo pipefail
CFG=$(realpath "$1"); TAG=$2; shift 2
SRC=/fsx/amine_dirhoussi/stable-bench-src/prime-rl
VENV=/fsx/amine_dirhoussi/envs/bench-prime-rl
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
RUN_ID=prime-rl-${TAG}-${SLURM_JOB_ID}
OUT=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/$RUN_ID
AUX=$OUT-aux # sidecar files (gpu.csv, job times): rl refuses to start in a non-empty run dir
mkdir -p "$AUX"

source "$VENV/bin/activate"
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONPATH=$BENCH/configs/prime-rl${PYTHONPATH:+:$PYTHONPATH}
export WANDB_PROJECT=stable-bench-prime-rl
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 \
    | sed -u "1s/\$/, node/; 2,\$s/\$/, $(hostname)/" > "$AUX/gpu.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

echo "job_start $(date +%s.%N) run_id=$RUN_ID node=$(hostname) port=$PORT cfg=$CFG" | tee "$AUX/job_times.txt"
cd "$SRC"
rl @ "$CFG" \
    --output-dir /fsx/amine_dirhoussi/stable-bench-runs/prime-rl \
    --run.name "$RUN_ID" \
    --inference.server.port "$PORT" \
    --orchestrator.model.client.base-url "http://localhost:$PORT/v1" \
    --rollout-transport.type zmq --rollout-transport.port $((PORT + 3)) \
    --no-monitors.prime \
    "$@"
rc=$?
echo "job_end $(date +%s.%N) rc=$rc" | tee -a "$AUX/job_times.txt"
exit $rc
