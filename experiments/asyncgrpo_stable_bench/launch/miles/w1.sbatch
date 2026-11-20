#!/bin/bash -l
#SBATCH --job-name=miles-w1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
# W1 math on miles (Megatron + SGLang, fully async), one node: 4 SGLang engines (TP=1) + 4 Megatron GPUs (TP=2, DP=2).
# Submit: b200-queue launch/miles/w1.sbatch   (env ARM=shared|recommended, SIZE=S|M, STEPS=200 (or MAX_STEPS), SAVE_STEPS=50)
# Config: configs/miles/w1-$SIZE[-recommended].args (one flag per line), knob mapping in configs/miles/EQUIVALENCE.md.
# SAVE_STEPS overrides the config's --save-interval (smokes); miles also always saves after the last step.
# EXTRA_ARGS="--flag value ..." is appended after the config (smokes of other shipped options).
set -uo pipefail
SIZE=${SIZE:-S}; ARM=${ARM:-shared}; TAG=${TAG:-$ARM}; STEPS=${STEPS:-${MAX_STEPS:-200}}; SAVE_STEPS=${SAVE_STEPS:-50}
case $SIZE in S) MODEL=Qwen3.5-2B ;; M) MODEL=Qwen3.5-9B ;; *) echo "unknown SIZE $SIZE"; exit 1 ;; esac
MT=qwen3.5-${MODEL#Qwen3.5-}
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.1.3.sqsh
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
CONFIG=$BENCH/configs/miles/w1-$SIZE$([ $ARM = shared ] || echo -$ARM).args
RUN_ID=miles-w1-$SIZE-$TAG-$SLURM_JOB_ID
RUN=/fsx/amine_dirhoussi/stable-bench-runs/miles/$RUN_ID
mkdir -p $RUN
cp $CONFIG $RUN/config.args
echo "RUN_ID $RUN_ID  RUN $RUN  NODE $(hostname)"
echo "JOB_START $(date +%s)" | tee $RUN/times.txt

# gpu.csv per SCHEMA: 1 Hz nvidia-smi on the host, plus a node column.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 \
    | awk -v n=$(hostname) 'NR==1{print $0", node"; fflush(); next}{print $0", "n; fflush()}' > $RUN/gpu.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

# b200-queue submits from a -c 1 --mem=2G waiter with --export=ALL: its SLURM_MEM_PER_NODE / SLURM_CPUS_PER_TASK leak in
# and would cap this srun step at 2 GiB and 1 CPU. Drop them and give the step the whole allocation.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG --container-mounts=/fsx:/fsx --container-workdir=/root/miles \
    --export=ALL,RUN=$RUN,CONFIG=$CONFIG,STEPS=$STEPS,SAVE_STEPS=$SAVE_STEPS,MODEL=$MODEL,MT=$MT,PORT_BASE=$((11000 + (SLURM_JOB_ID % 90) * 10)) \
    bash -c '
set -uo pipefail
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles/common.sh
{ git -C /root/miles log -1 --format="miles %H"; git -C /root/Megatron-LM log -1 --format="megatron %H"; git -C /sgl-workspace/sglang log -1 --format="sglang %H";
  python3 -c "import torch, sglang, fla, causal_conv1d, transformers; print(\"torch\", torch.__version__, \"sglang\", sglang.__version__, \"fla\", fla.__version__, \"causal_conv1d\", causal_conv1d.__version__, \"transformers\", transformers.__version__)"; } | tee $RUN/versions.txt
convert_ckpt $MT $MODELS/$MODEL $CKPT/${MODEL}_torch_dist 4 || exit 1
start_ray 8
echo "TRAIN_START $(date +%s)" >> $RUN/times.txt
# config file: one flag (with its value) per line, # comments allowed; RUN and STEPS are substituted.
ARGS=$(grep -v "^\s*#" $CONFIG | sed -e "s|\$RUN|$RUN|g" -e "s|\$STEPS|$STEPS|g" -e "s|^--save-interval 50$|--save-interval $SAVE_STEPS|" | tr "\n" " ")
# EXTRA_ARGS (smokes): appended last, so a repeated flag overrides the config (argparse keeps the last value).
python3 train_async.py $(model_args $MT) $ARGS ${EXTRA_ARGS:-} > $RUN/train.log 2>&1
rc=$?
echo "TRAIN_END $(date +%s) EXIT $rc" >> $RUN/times.txt
cp -r /tmp/ray/session_latest/logs $RUN/raylogs 2>/dev/null
stop_ray
# The HF export holds the trained text weights only (miles never trains or syncs the vision tower). Add the frozen
# vision weights back from the base checkpoint so the export loads as the original Qwen3_5ForConditionalGeneration.
for d in $RUN/hf/step_*; do python3 $BENCH/configs/miles/complete_hf_export.py $MODELS/$MODEL $d; done
# workloads/math/eval.sbatch reads <run>/checkpoint-<step>; miles names exports by rollout id (step = rollout_id + 1).
for d in $RUN/hf/step_*; do r=${d##*_}; ln -sfn hf/step_$r $RUN/checkpoint-$((r + 1)); done
exit $rc
'
rc=$?
echo "JOB_END $(date +%s) EXIT $rc" | tee -a $RUN/times.txt
exit $rc
