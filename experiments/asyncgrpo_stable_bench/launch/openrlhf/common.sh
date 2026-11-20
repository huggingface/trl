# Sourced by every OpenRLHF sbatch script: venv, caches, a private single-node Ray head, and the 1 Hz GPU sampler.
# Needs RUN_DIR (output dir for this run) set by the caller.
VENV=/fsx/amine_dirhoussi/envs/bench-openrlhf
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
source "$VENV/bin/activate"
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH   # DeepSpeed JIT-builds its fused Adam
export TORCH_EXTENSIONS_DIR=/fsx/amine_dirhoussi/.cache/torch_extensions_openrlhf
# Trainer ranks hit the caching allocator's OOM-and-retry path on ~16 GB fp32 logits of 16k-token micro-batches
# (NOTES.md item 20); expandable segments stop the fragmentation. Exported before `ray start` so every actor gets it.
# Set PYTORCH_CUDA_ALLOC_CONF= (empty) to run with the PyTorch default.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_DATASETS_CACHE=/fsx/amine_dirhoussi/.cache/hf-datasets-openrlhf
mkdir -p "$RUN_DIR"
NGPU=$(nvidia-smi -L | wc -l)

# Nodes are shared between jobs (and other agents run Ray as the same user), so: own temp dir, ports derived from
# the job id, and no `ray stop` at exit (it kills every Ray process of the user on the node). Slurm's cgroup cleans up.
# Start from the job-id slot and move to the next 1000-port block if any fixed port in it is taken (job 364294 died
# with "Failed to start GCS" when another process on the shared node held its GCS port).
for k in $(seq 0 29); do
    BASE=$((30000 + ((SLURM_JOB_ID + k) % 30) * 1000))
    python3 -c "
import socket, sys
for p in [$BASE + i for i in range(8)] + [$BASE + 50] + list(range($BASE + 100, $BASE + 1000, 50)):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: s.bind(('', p))
    except OSError: sys.exit(1)
    finally: s.close()
" && break
done
echo "ray port block $BASE"
export VLLM_PORT=$((BASE + 50))
RAY_TMP=/tmp/ray_${SLURM_JOB_ID}
IP=$(hostname -I | awk '{print $1}')
ray start --head --node-ip-address="$IP" --port=$BASE --include-dashboard=false --disable-usage-stats \
    --ray-client-server-port=$((BASE + 1)) --node-manager-port=$((BASE + 2)) --object-manager-port=$((BASE + 3)) \
    --dashboard-agent-grpc-port=$((BASE + 4)) --dashboard-agent-listen-port=$((BASE + 5)) \
    --metrics-export-port=$((BASE + 6)) --runtime-env-agent-port=$((BASE + 7)) \
    --min-worker-port=$((BASE + 100)) --max-worker-port=$((BASE + 999)) \
    --num-gpus="$NGPU" --num-cpus="${SLURM_CPUS_ON_NODE:-32}" --object-store-memory=$((40 * 1024 ** 3)) \
    --temp-dir="$RAY_TMP"
export RAY_ADDRESS=$IP:$BASE

# gpu.csv per collect/SCHEMA.md: 1 Hz nvidia-smi on every node plus a node column.
echo "node,timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]" > "$RUN_DIR/gpu.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 \
    | sed -u "s/^/$(hostname),/" >> "$RUN_DIR/gpu.csv" &
GPU_SAMPLER=$!
trap 'kill $GPU_SAMPLER 2>/dev/null' EXIT
echo "JOB_START $(date +%s.%N)" > "$RUN_DIR/job_times.txt"
