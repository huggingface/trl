# Sourced by every verl sbatch script. Expects RUN_DIR to be set; creates it, activates the venv,
# starts a private single-node Ray head (job-derived ports and temp dir, so jobs sharing a node don't clash)
# and the 1 Hz gpu.csv sampler from collect/SCHEMA.md. Everything is torn down on exit.

VENV=/fsx/amine_dirhoussi/envs/bench-verl
SRC=/fsx/amine_dirhoussi/stable-bench-src/verl
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export HF_HUB_OFFLINE=1
export VERL_USE_UV=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# TransferQueue: storage managers wait this long for the controller to ACK a data-status update, then only log an error
# and continue; the data is never marked ready and a later training shard comes back empty (KeyError loss_mask). The 30 s
# default is too short after a validation pass on 8 B200 (envs/verl/NOTES.md). Ray workers inherit it from `ray start`.
export TQ_DATA_UPDATE_RESPONSE_TIMEOUT=600

mkdir -p "$RUN_DIR"
source "$VENV/bin/activate"
cd "$SRC"
echo "job $SLURM_JOB_ID on $(hostname) gpus=$CUDA_VISIBLE_DEVICES start $(date -u +%FT%TZ)"
echo "JOB_START_EPOCH $(date +%s.%N)"

# gpu.csv: every GPU of the node (nvidia-smi sees the whole node), plus a node column.
NODE=$(hostname)
( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 \
  | sed -u "s/$/, $NODE/" > "$RUN_DIR/gpu.csv.body" ) &
GPU_SAMPLER=$!

# Private Ray head. Ports are spaced by job id; temp dir is short (Ray's unix socket paths have a length limit).
# 1000 ports per job: Ray prestarts one worker per CPU (92 on an exclusive node), each needing its own worker port.
BASE=$((20000 + (SLURM_JOB_ID % 40) * 1000))
export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID
mkdir -p "$RAY_TMPDIR"
NGPU=$(nvidia-smi -L | wc -l)
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
ray start --head --port=$BASE --node-manager-port=$((BASE+1)) --object-manager-port=$((BASE+2)) \
  --ray-client-server-port=$((BASE+3)) --dashboard-port=$((BASE+4)) --dashboard-agent-listen-port=$((BASE+5)) \
  --dashboard-agent-grpc-port=$((BASE+6)) --metrics-export-port=$((BASE+7)) --runtime-env-agent-port=$((BASE+8)) \
  --min-worker-port=$((BASE+100)) --max-worker-port=$((BASE+999)) \
  --temp-dir="$RAY_TMPDIR" --num-gpus="$NGPU" --num-cpus="${SLURM_CPUS_ON_NODE:-32}" --include-dashboard=false \
  --disable-usage-stats >/dev/null
export RAY_ADDRESS=127.0.0.1:$BASE

cleanup() {
  [ -n "${CLEANED:-}" ] && return; CLEANED=1
  ray stop --force >/dev/null 2>&1 || true
  kill "$GPU_SAMPLER" 2>/dev/null; pkill -P "$GPU_SAMPLER" 2>/dev/null
  { echo "timestamp, index, utilization.gpu [%], memory.used [MiB], power.draw [W], node"; cat "$RUN_DIR/gpu.csv.body"; } > "$RUN_DIR/gpu.csv" && rm -f "$RUN_DIR/gpu.csv.body"
  rm -rf "$RAY_TMPDIR"
  echo "JOB_END_EPOCH $(date +%s.%N)"
}
trap cleanup EXIT
