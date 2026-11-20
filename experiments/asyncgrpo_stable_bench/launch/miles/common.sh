# Sourced INSIDE the miles container (Pyxis). Helpers shared by the miles launch scripts.
# We do not use miles' Python launchers (scripts/run_*.py -> command_utils): their Ray backend runs
# `pkill -9 sglang; ray stop --force; pkill -9 ray; pkill -9 miles; pkill -9 redis` before starting, and Enroot
# shares the host PID namespace, so on a shared node that kills other jobs' Ray/SGLang processes (same user).
# Instead we start a private Ray head on job-derived ports and run train*.py directly with the env vars the
# launcher would have put in the Ray runtime env (command_utils/common.py:train_env_vars + ray_backend._ray_env_vars).

MILES=/root/miles
MEGATRON=/root/Megatron-LM
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
# configs/miles holds w1_reward.py, workloads/math holds the shared reward.py it imports.
export PYTHONPATH=$MILES:$MEGATRON:$BENCH/configs/miles:$BENCH/workloads/math${PYTHONPATH:+:$PYTHONPATH}
# FA4 window fix (configs/miles/fa4_window_fix/sitecustomize.py, see envs/miles/NOTES.md): on by default, FA4_WINDOW_FIX=0 disables.
[ "${FA4_WINDOW_FIX:-0}" = 1 ] && export PYTHONPATH=$BENCH/configs/miles/fa4_window_fix:$PYTHONPATH
# math-verify is not in the image: use the uv overlay (envs/miles/prep_w1.sh) only when the image lacks it.
python3 -c "import math_verify" 2>/dev/null || export PYTHONPATH=$PYTHONPATH:/fsx/amine_dirhoussi/envs/bench-miles
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=1
export MASTER_ADDR=127.0.0.1 no_proxy=127.0.0.1
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
MODELS=/fsx/amine_dirhoussi/stable-bench-runs/miles/models
DATA=/fsx/amine_dirhoussi/stable-bench-runs/miles/datasets
CKPT=/fsx/amine_dirhoussi/stable-bench-runs/miles/ckpt

# Megatron model args. 2B is not shipped by miles: it is scripts/models/qwen3.5-0.8B.py with the 2B widths
# (Hub config: hidden 2048, intermediate 6144; everything else identical to 0.8B).
model_args() {
    case "$1" in
        qwen3.5-2B) echo "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec --disable-bias-linear --qk-layernorm \
--group-query-attention --num-attention-heads 8 --num-query-groups 2 --kv-channels 256 --num-layers 24 \
--hidden-size 2048 --ffn-hidden-size 6144 --normalization RMSNorm --apply-layernorm-1p \
--position-embedding-type rope --norm-epsilon 1e-6 --rotary-percent 0.25 --swiglu --vocab-size 248320 \
--rotary-base 10000000 --attention-output-gate" ;;
        *) python3 $MILES/miles/utils/external_utils/model_args_utils.py "$1" ;;
    esac
}

# convert_ckpt <megatron model type> <hf dir> <dst> <nproc>: HF -> torch_dist, skipped if already done
# (same command as command_utils.convert_checkpoint).
convert_ckpt() {
    local mt=$1 hf=$2 dst=$3 n=$4
    if [ "$(cat $dst/latest_checkpointed_iteration.txt 2>/dev/null)" = release ]; then echo "convert: $dst exists"; return 0; fi
    local t0=$(date +%s)
    # fh_wrap.py dumps every rank's Python stack every CONVERT_STACK_EVERY s while it runs (no ptrace on the nodes).
    torchrun --nproc-per-node $n --master-port $((PORT_BASE + 9)) $BENCH/launch/miles/fh_wrap.py \
        ${CONVERT_STACK_EVERY:-600} $MILES/tools/convert_hf_to_torch_dist.py \
        $(model_args $mt) --hf-checkpoint $hf --save $dst || return 1
    echo "CONVERT_WALL_S $mt $(( $(date +%s) - t0 ))"
}

# Private Ray head. PORT_BASE comes from the job id so co-scheduled jobs don't collide.
start_ray() {
    local ngpu=$1
    ray start --head --node-ip-address 127.0.0.1 --num-gpus $ngpu --num-cpus ${SLURM_CPUS_ON_NODE:-16} \
        --port $PORT_BASE --node-manager-port $((PORT_BASE + 1)) --object-manager-port $((PORT_BASE + 2)) \
        --ray-client-server-port $((PORT_BASE + 3)) --dashboard-agent-listen-port $((PORT_BASE + 4)) \
        --dashboard-agent-grpc-port $((PORT_BASE + 5)) --metrics-export-port $((PORT_BASE + 6)) \
        --runtime-env-agent-port $((PORT_BASE + 7)) --dashboard-port $((PORT_BASE + 8)) \
        --min-worker-port 0 --max-worker-port 0 --temp-dir /tmp/ray --disable-usage-stats
    export RAY_ADDRESS=127.0.0.1:$PORT_BASE
}

stop_ray() {
    ray stop --grace-period 30 >/dev/null 2>&1 || true
}
