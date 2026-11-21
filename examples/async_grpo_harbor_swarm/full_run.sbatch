#!/bin/bash
#SBATCH --job-name=trl-swarm-full-16agents
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --mem=0
#SBATCH --time=08:00:00
set -euo pipefail
repo=${SWARM_REPO:?}
root=${SWARM_RUN_DIR:?}
# Trainer and vLLM run in separate host/container namespaces. Use socket transport
# for this retry instead of CUDA IPC/shared-memory paths across that boundary.
export NCCL_NET=Socket NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1
mkdir -p "$root"
printf '%s\n' "$SLURM_JOB_ID" > "$root/job"
printf '%s\n' "$(( $(date +%s) + ${SWARM_RUN_SECONDS:-28800} ))" > "$root/deadline"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
server_pid=
monitor_pid=
cleanup() {
 [ -z "$server_pid" ] || kill "$server_pid" 2>/dev/null || true
 [ -z "$monitor_pid" ] || kill "$monitor_pid" 2>/dev/null || true
 [ -z "$monitor_pid" ] || wait "$monitor_pid" || true
}
trap cleanup EXIT
"$repo/.venv/bin/python" "$repo/examples/async_grpo_harbor_swarm/monitor_full.py" --run-dir "$root" --job "$SLURM_JOB_ID" > "$root/monitor.log" 2>&1 & monitor_pid=$!
start_server() {
 srun --exact --exclusive -N1 -n1 -w "${nodes[0]}" --gpus=4 --gpus-per-node=4 --cpus-per-task=24 --mem=512G --container-image=/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh --no-container-mount-home --container-mounts="/fsx:/fsx" bash "$repo/examples/async_grpo_harbor_swarm/full_server.sh" > "$root/server-$1.log" 2>&1 & server_pid=$!
 "$repo/.venv/bin/python" - "$root/server-url" <<'PYWAIT'
import pathlib,time,sys,urllib.request
for _ in range(240):
 try:
  url=pathlib.Path(sys.argv[1]).read_text().strip()
  urllib.request.urlopen(url+'/health',timeout=3);break
 except Exception:time.sleep(5)
else:raise SystemExit('vLLM startup failed')
PYWAIT
}
if [[ "${SWARM_VALIDATE:-0}" == 1 ]]; then
start_server smoke
srun --exact --exclusive -N1 -n1 -w "${nodes[0]}" --gpus=4 --gpus-per-node=4 --cpus-per-task=64 --mem=1400G bash "$repo/examples/async_grpo_harbor_swarm/full_node.sh" smoke > "$root/smoke.log" 2>&1
kill "$server_pid"
wait "$server_pid" || true
server_pid=
rm -f "$root/server-url"
fi
start_server train
srun --exact --exclusive -N1 -n1 -w "${nodes[0]}" --gpus=4 --gpus-per-node=4 --cpus-per-task=64 --mem=1400G bash "$repo/examples/async_grpo_harbor_swarm/full_node.sh" experiment > "$root/experiment.log" 2>&1
