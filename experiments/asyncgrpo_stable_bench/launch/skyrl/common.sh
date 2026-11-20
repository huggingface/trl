# Sourced by every SkyRL sbatch script. Sets up the venv, caches, a private Ray head and the gpu.csv sampler.
# Needs: RUN_DIR (output dir for this run) set before sourcing.

export SKYRL_SRC=/fsx/amine_dirhoussi/stable-bench-src/skyrl
export SKYRL_VENV=${SKYRL_VENV:-/fsx/amine_dirhoussi/envs/bench-skyrl}
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
# Ray workers inherit the driver's interpreter; never let the uv runtime-env hook re-resolve an env per worker.
unset RAY_RUNTIME_ENV_HOOK VIRTUAL_ENV
source "$SKYRL_VENV/bin/activate"
export PATH="$SKYRL_VENV/bin:$PATH"
cd "$SKYRL_SRC"

mkdir -p "$RUN_DIR"

# Private Ray head: own temp dir (node-local, short path for unix sockets) and a 1000-port block derived from the job id,
# so co-scheduled jobs on a shared node never attach to each other's cluster. 1000 ports because Ray prestarts one
# worker per CPU, each on its own port. GPUs, CPUs and object store are passed explicitly.
_ray_base=${RAY_BASE:-$((21500 + (SLURM_JOB_ID % 37) * 1000))}
export RAY_TMP=/tmp/ray-sk-$SLURM_JOB_ID
export RAY_PORT=$_ray_base
_ip=$(hostname -I | awk '{print $1}')
_ngpu=$(nvidia-smi -L | wc -l)
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && _ngpu=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
ray start --head --disable-usage-stats --include-dashboard=false \
    --node-ip-address="$_ip" \
    --temp-dir="$RAY_TMP" \
    --port=$RAY_PORT \
    --node-manager-port=$((_ray_base + 1)) \
    --object-manager-port=$((_ray_base + 2)) \
    --ray-client-server-port=$((_ray_base + 3)) \
    --dashboard-agent-listen-port=$((_ray_base + 5)) \
    --dashboard-agent-grpc-port=$((_ray_base + 6)) \
    --metrics-export-port=$((_ray_base + 7)) \
    --runtime-env-agent-port=$((_ray_base + 8)) \
    --min-worker-port=$((_ray_base + 100)) \
    --max-worker-port=$((_ray_base + 999)) \
    --num-gpus="$_ngpu" \
    --num-cpus="${SLURM_CPUS_ON_NODE:-32}" \
    --object-store-memory=$((40 * 1024 ** 3)) >"$RUN_DIR/ray_start.log" 2>&1 || {
        cat "$RUN_DIR/ray_start.log"; cp -r "$RAY_TMP/session_latest/logs" "$RUN_DIR/ray_logs" 2>/dev/null; exit 1; }
export RAY_ADDRESS=$_ip:$RAY_PORT

# 1 Hz GPU sampler (collect/SCHEMA.md gpu.csv), with a node column.
_node=$(hostname)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 \
    | awk -v n="$_node" 'BEGIN{print "timestamp,index,utilization.gpu,memory.used,power.draw,node"} {print $0", "n; fflush()}' \
    >"$RUN_DIR/gpu.csv" &
_smi_pid=$!

skyrl_cleanup() {
    local rc=$?
    # Keep Ray's logs (node-local /tmp) when the job fails.
    [ $rc -ne 0 ] && cp -r "$RAY_TMP/session_latest/logs" "$RUN_DIR/ray_logs" 2>/dev/null
    # No `ray stop`: it kills every Ray process of this user on the node, including other jobs' clusters.
    # Slurm reaps this job's Ray daemons with the job cgroup.
    kill $_smi_pid 2>/dev/null
    echo "JOB_END=$(date +%s.%N)" >>"$RUN_DIR/job_times.txt"
}
trap skyrl_cleanup EXIT
echo "JOB_START=$(date +%s.%N)" >"$RUN_DIR/job_times.txt"
