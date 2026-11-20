#!/bin/bash
set -euo pipefail
repo=${SWARM_REPO:?}
root=${SWARM_RUN_DIR:?}
run="$root/$1"
mkdir -p "$run"
export VENV="$repo/.venv"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 COMPOSE_PARALLEL_LIMIT=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
local_root=$(mktemp -d /tmp/trl-swarm.XXXXXX)
export XDG_RUNTIME_DIR=$local_root/runtime
export HF_XET_CACHE=$local_root/xet
export TRITON_CACHE_DIR=$local_root/triton
export LITELLM_LOCAL_MODEL_COST_MAP=True
export VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
mkdir -p "$XDG_RUNTIME_DIR" "$local_root/bin" "$local_root/network" "$local_root/model" "$local_root/vllm"
bundle=${SWARM_PODMAN_BUNDLE:?}
export CONTAINERS_CONF=$local_root/containers.conf
cat > "$CONTAINERS_CONF" <<EOF
[containers]
seccomp_profile = "$bundle/etc/containers/seccomp.json"
[engine]
helper_binaries_dir = ["$bundle/usr/local/lib/podman", "$bundle/usr/local/bin", "/usr/lib/podman"]
conmon_path = ["$bundle/usr/local/lib/podman/conmon"]
runtime = "crun"
[engine.runtimes]
crun = ["$bundle/usr/local/bin/crun"]
[network]
network_backend = "netavark"
default_rootless_network_cmd = "slirp4netns"
EOF
export CONTAINERS_CONF_OVERRIDE=$CONTAINERS_CONF
export CONTAINERS_REGISTRIES_CONF=$local_root/registries.conf
printf 'unqualified-search-registries = ["docker.io"]\n' > "$CONTAINERS_REGISTRIES_CONF"
cat > "$local_root/bin/podman" <<EOF
#!/bin/bash
exec "$bundle/usr/local/bin/podman" --root "$local_root/graphroot" --runroot "$local_root/runroot" --storage-driver overlay --storage-opt overlay.mount_program="$bundle/usr/local/bin/fuse-overlayfs" --cgroup-manager cgroupfs --events-backend file "\$@"
EOF
cat > "$local_root/bin/systemd-run" <<EOF
#!/bin/bash
if [[ "\${1:-} \${2:-} \${3:-}" == "-q --scope --user" && "\${4:-}" == "$bundle/usr/local/lib/podman/aardvark-dns" ]]; then shift 3; exec "\$@"; fi
exec /usr/bin/systemd-run "\$@"
EOF
cp "$repo/examples/async_grpo_harbor_swarm/podman_compose_compat.py" "$local_root/bin/podman-compose"
chmod 700 "$local_root/bin/"*
export PATH="$local_root/bin:$VENV/bin:$bundle/usr/local/bin:/usr/local/bin:/usr/bin:/bin"
export PODMAN_COMPOSE_PROVIDER=$local_root/bin/podman-compose
unset CONTAINER_HOST CONTAINER_CONNECTION DOCKER_HOST
printf '%s\n' "$local_root" > "$run/local-runtime-dir"
if [[ -n "${SWARM_AGENT_IMAGE_ARCHIVE:-}" ]]; then
 podman load -i "$SWARM_AGENT_IMAGE_ARCHIVE" > "$run/agent-image-load.log" 2>&1
fi
health_pid= server_pid= train_pid=
cleanup() {
  [ -z "$train_pid" ] || kill "$train_pid" 2>/dev/null || true
  [ ! -d "$local_root/trackio" ] || cp -a "$local_root/trackio" "$run/trackio-local"
  [ -z "$health_pid" ] || kill "$health_pid" 2>/dev/null || true
  [ -z "$server_pid" ] || kill "$server_pid" 2>/dev/null || true
  timeout 60 "$local_root/bin/podman" rm --all --force --volumes >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$repo"
python examples/async_grpo_harbor_swarm/podman_healthchecks.py > "$run/healthchecks.log" 2>&1 & health_pid=$!
export TRACKIO_DIR="$local_root/trackio"
export SWARM_GATEWAY_TOKEN=$(python -c 'import secrets; print(secrets.token_hex(32))')
url=$(cat "$root/server-url")
host=$(hostname -I | awk '{print $1}')
port=$((31000 + SLURM_JOB_ID % 10000))
gateway_port=$((port + 1))
gateway="http://$host:$gateway_port"
python examples/async_grpo_harbor_swarm/policy_gateway.py --upstream "$url" --output-dir "$run" --port "$gateway_port" > "$run/gateway.log" 2>&1 & server_pid=$!
python - "$gateway" <<'PYWAIT'
import sys,time,urllib.request
for _ in range(120):
 try:
  urllib.request.urlopen(sys.argv[1]+'/status', timeout=3);break
 except Exception: time.sleep(1)
else: raise SystemExit('gateway startup failed')
PYWAIT
run_training() {
 "$@" & train_pid=$!
 local finished_pid
 if ! wait -n -p finished_pid "$train_pid" "$health_pid" "$server_pid"; then
  echo 'Training or required runtime service failed' >&2
  return 1
 fi
 [[ "$finished_pid" == "$train_pid" ]] || { echo 'Required runtime service exited during training' >&2; return 1; }
 train_pid=
}
common=(--vllm-url "$url" --gateway-url "$gateway" --capture-url "http://$host:$port" --capture-port "$port" --difficulties 10 15 --agents 16 --num-generations 16 --max-staleness 4 --agent-timeout 1800 --max-turns 64 --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --full-finetune)
if [[ "$1" == smoke ]]; then
 run_training accelerate launch --config_file trl/accelerate_configs/fsdp2.yaml --num_processes 4 --main_process_port 29551 examples/async_grpo_harbor_swarm/async_grpo_harbor_swarm.py train "${common[@]}" --output-dir "$run/train" --synthetic-smoke --require-nonzero-update --max-steps 2 --teams-per-update 16 --concurrent-teams 16 --max-groups 0 --num-tasks 2 --token-budget 16384 > "$run/train.log" 2>&1
 python examples/async_grpo_harbor_swarm/verify_full_checkpoint.py "$run/train"
 exit
fi
concurrency=16
if [[ "${SWARM_VALIDATE:-0}" == 1 ]]; then
concurrency=0
for level in 4 8 16; do
 if python examples/async_grpo_harbor_swarm/async_grpo_harbor_swarm.py evaluate "${common[@]}" --output-dir "$run/gate-$level" --num-tasks "$level" --concurrent-teams "$level" > "$run/gate-$level.log" 2>&1 && python - "$run/gate-$level/evaluation.json" "$level" <<'PYGATE'
import json,sys
r=json.load(open(sys.argv[1])); assert r['teams']==r['scored']==int(sys.argv[2]) and r['unscorable']==0,r
PYGATE
 then concurrency=$level; else break; fi
 done
[[ "$concurrency" -gt 0 ]] || { echo 'Initial native concurrency gate failed'; exit 1; }
fi
printf '%s\n' "$concurrency" > "$root/concurrent-teams"
end=$(cat "$root/deadline")
admission=$((end - 2700))
collection=$((end - 600))
budget=$((end - $(date +%s) - 600))
[[ "$budget" -gt 3600 ]] || { echo 'Insufficient allocation remains for training'; exit 1; }
run_training accelerate launch --config_file trl/accelerate_configs/fsdp2.yaml --num_processes 4 --main_process_port 29551 examples/async_grpo_harbor_swarm/async_grpo_harbor_swarm.py train "${common[@]}" --output-dir "$run/train" --max-steps 10000 --max-groups 0 --teams-per-update 64 --concurrent-teams "$concurrency" --admission-deadline "$admission" --collection-deadline "$collection" --time-limit-seconds "$budget" --report-to trackio --trackio-space-id edbeeching/trl-harbor-swarm-trackio --run-name "qwen35-4b-full-16agents-$SLURM_JOB_ID" > "$run/train.log" 2>&1
