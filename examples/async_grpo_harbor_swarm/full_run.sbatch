#!/bin/bash
#SBATCH --job-name=trl-swarm-full-16agents
#SBATCH --partition=hopper-prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --mem=0
#SBATCH --time=08:00:00
set -euo pipefail
repo=${SWARM_REPO:?}
root=${SWARM_RUN_DIR:?}
mkdir -p "$root"
printf '%s\n' "$SLURM_JOB_ID" > "$root/job"
printf '%s\n' "$(( $(date +%s) + 28800 ))" > "$root/deadline"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
server_pid=
cleanup() {
 [ -z "$server_pid" ] || kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
start_server() {
 srun --exclusive -N1 -n1 -w "${nodes[1]}" --gpus=8 --cpus-per-task=88 --container-image=/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh --no-container-mount-home --container-mounts="/fsx:/fsx" bash "$repo/examples/async_grpo_harbor_swarm/full_server.sh" > "$root/server-$1.log" 2>&1 & server_pid=$!
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
start_server smoke
srun --exclusive -N1 -n1 -w "${nodes[0]}" --gpus=8 --cpus-per-task=88 bash "$repo/examples/async_grpo_harbor_swarm/full_node.sh" smoke > "$root/smoke.log" 2>&1
kill "$server_pid"
wait "$server_pid" || true
server_pid=
rm -f "$root/server-url"
start_server train
srun --exclusive -N1 -n1 -w "${nodes[0]}" --gpus=8 --cpus-per-task=88 bash "$repo/examples/async_grpo_harbor_swarm/full_node.sh" experiment > "$root/experiment.log" 2>&1
