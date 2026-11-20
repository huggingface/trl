#!/bin/bash
# Text-to-SQL on SkyRL (Megatron + LoRA, Qwen3.5-35B-A3B), 1 node exclusive, colocated sync (configs/skyrl/t2sql-35b.overrides).
# Submit from the bench dir:
#   PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --nodes=1 --gres=gpu:8 --exclusive -t 16:00:00 launch/skyrl/t2sql.sbatch
# Installs the Megatron venv first if it is missing (envs/skyrl/install_megatron.sh, on this node: hopper-cpu was full).
# In-job watcher: launch/skyrl/t2sql_watch.py (cancels the job on failure). Env: EXTRA (extra key=value overrides).
#SBATCH --job-name=skyrl-t2sql
#SBATCH --nodes=1
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
set -uo pipefail

BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
CONFIG=${CONFIG:-$BENCH/configs/skyrl/t2sql-35b.overrides}
DATA=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/t2sql/train.parquet
export SKYRL_VENV=/fsx/amine_dirhoussi/envs/bench-skyrl-mcore
# RUN_ID set by resume_helper.sh to resume the same run; RUN_SUFFIX e.g. -8gpu
RUN_ID=${RUN_ID:-skyrl-t2sql-Qwen3.5-35B-A3B-lora32${RUN_SUFFIX:-}-$SLURM_JOB_ID}
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/skyrl/$RUN_ID
mkdir -p $RUN_DIR
echo "$(date -u) job $SLURM_JOB_ID on $(hostname)" | tee -a $RUN_DIR/job.txt

# TE 2.19 dlopens libcudnn_graph.so.9 by name; the pip cuDNN lives in the venv (SkyRL's Docker image has a system cuDNN)
export LD_LIBRARY_PATH=$SKYRL_VENV/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [ ! -f $SKYRL_VENV/.installed ]; then
    echo "$(date -u) installing $SKYRL_VENV" | tee -a $RUN_DIR/job.txt
    flock /fsx/amine_dirhoussi/envs/.bench-skyrl-mcore.lock bash -c "[ -f $SKYRL_VENV/.installed ] || { bash $BENCH/envs/skyrl/install_megatron.sh > $RUN_DIR/install.log 2>&1 && touch $SKYRL_VENV/.installed; }"
    [ -f $SKYRL_VENV/.installed ] || { echo "install failed"; tail -n 40 $RUN_DIR/install.log; exit 1; }
    echo "$(date -u) installed: $(grep INSTALL_WALL_S $RUN_DIR/install.log)" | tee -a $RUN_DIR/job.txt
fi
for i in $(seq 120); do [ -f $DATA ] && break; sleep 30; done
[ -f $DATA ] || { echo "no $DATA"; exit 1; }

# Blackwell flags from SkyRL's Qwen3.5-35B-A3B recipe: triton MoE in vLLM (flashinfer trtllm bf16 needs intermediate % 128),
# triton GDN backend in fla (TileLang aborts in the packed backward)
export VLLM_USE_FLASHINFER_MOE_FP16=0 FLA_TILELANG=0
export HF_HUB_OFFLINE=1
# SkyRL's vLLM health wait (default 600 s): job 373719's engine took 9m47s to come up on a shared node (5 min of
# process start/imports from /fsx before weight loading), and the wait expired seconds before it was ready
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1800
export TENSORBOARD_DIR=$RUN_DIR/tb
export T2SQL_EPISODES_FILE=$RUN_DIR/episodes.jsonl
S=/scratch/$USER/skyrl-t2sql-$SLURM_JOB_ID; mkdir -p -m 700 $S/snapshots $S/tmp
export TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$S/snapshots VLLM_CACHE_ROOT=$S/vllm-cache
# per-job compile caches on node-local disk (skyrl-026: a TP8 worker died in inductor autotune with "CUDA driver error: file not found")
export TORCHINDUCTOR_CACHE_DIR=$S/inductor TRITON_CACHE_DIR=$S/triton
# The env module, the episode port and River's recipe must be importable in every Ray worker (inherited from the raylet).
export PYTHONPATH=$BENCH/launch/skyrl:$BENCH/workloads/text2sql:$D/river-recipes/text-to-sql/src${PYTHONPATH:+:$PYTHONPATH}
# Ray port block below the ephemeral range (32768+): job 373617's runtime-env agent lost its port 49508 to another
# process on the shared node. First 1000-port block in 12000-31999 with nothing listening.
for b in $(seq $((12000 + (SLURM_JOB_ID % 20) * 1000)) 1000 31000) $(seq 12000 1000 31000); do
    ss -Htan "sport >= :$b and sport <= :$((b + 999))" | grep -q . || { export RAY_BASE=$b; break; }
done
echo "RAY_BASE=$RAY_BASE" | tee -a $RUN_DIR/job.txt
source $BENCH/launch/skyrl/common.sh
trap 'skyrl_cleanup; rm -rf $S' EXIT

mapfile -t ARGS < <(grep -v '^\s*#' "$CONFIG" | grep -v '^\s*$')
cp "$CONFIG" "$RUN_DIR/config.overrides"
echo "RUN_ID=$RUN_ID CONFIG=$CONFIG SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1) EXTRA=${EXTRA:-}" | tee "$RUN_DIR/run_info.txt"

python $BENCH/launch/skyrl/t2sql_watch.py $RUN_DIR &
WATCH=$!

# NGPU < 8 (shared node): same recipe on fewer GPUs, Megatron TP2 EP=NGPU, one vLLM engine TP=NGPU
NGPU=$(nvidia-smi -L | wc -l); [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "NGPU=$NGPU" | tee -a "$RUN_DIR/run_info.txt"
[ -n "${REQUIRE_NGPU:-}" ] && [ "$NGPU" != "$REQUIRE_NGPU" ] && { echo "NGPU=$NGPU but REQUIRE_NGPU=$REQUIRE_NGPU"; exit 1; }
python -m skyrl_t2sql_main \
  "${ARGS[@]}" \
  trainer.placement.policy_num_gpus_per_node=$NGPU \
  trainer.policy.megatron_config.expert_model_parallel_size=$NGPU \
  generator.inference_engine.tensor_parallel_size=$NGPU \
  data.train_data="['$DATA']" \
  data.val_data="[]" \
  trainer.project_name=asyncgrpo-stable-bench \
  trainer.run_name=$RUN_ID \
  trainer.log_path=$RUN_DIR/skyrl-logs \
  trainer.ckpt_path=$RUN_DIR/ckpts \
  trainer.export_path=$RUN_DIR \
  ${EXTRA:-} 2>&1 | tee "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
kill $WATCH 2>/dev/null
echo "EXIT=$status" >>"$RUN_DIR/run_info.txt"
exit $status
