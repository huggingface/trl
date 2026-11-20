#!/bin/bash -l
#SBATCH --job-name=miles-w1-moe
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
# W1 math on Qwen3.5-35B-A3B (MoE) with miles, one node: 4 SGLang engines (TP1) + 4 Megatron GPUs (TP1 x EP4).
# Submit: PRIO_TAG=PRIO_TRAIN b200-queue launch/miles/w1-moe.sbatch   (env TAG, STEPS (default 50), CONFIG_NAME (default w1-moe), MT, EXTRA_ARGS)
# Same skeleton as launch/miles/w1.sbatch, minus the torch_dist conversion and the HF export completion:
# the LoRA path (--megatron-to-hf-mode bridge) loads the HF checkpoint directly. MT=qwen3.5-35B-A3B + a config with --ref-load is the full-FT path.
set -uo pipefail
TAG=${TAG:-run}; STEPS=${STEPS:-50}; CONFIG_NAME=${CONFIG_NAME:-w1-moe}; MT=${MT:-qwen3.5-35B-A3B_lora}
MODEL=Qwen3.5-35B-A3B
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.1.3.sqsh
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
CONFIG=$BENCH/configs/miles/$CONFIG_NAME.args
RUN_ID=miles-w1-moe-$TAG-$SLURM_JOB_ID
RUN=/fsx/amine_dirhoussi/stable-bench-runs/miles/$RUN_ID
mkdir -p $RUN
cp $CONFIG $RUN/config.args
echo "EXTRA_ARGS ${EXTRA_ARGS:-}" > $RUN/extra_args.txt
echo "RUN_ID $RUN_ID  RUN $RUN  NODE $(hostname)"
echo "JOB_START $(date +%s)" | tee $RUN/times.txt

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 \
    | awk -v n=$(hostname) 'NR==1{print $0", node"; fflush(); next}{print $0", "n; fflush()}' > $RUN/gpu.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

# b200-queue's waiter leaks a 2 GiB / 1 CPU limit through --export=ALL (envs/miles/NOTES.md item 9).
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG --container-mounts=/fsx:/fsx --container-workdir=/root/miles \
    --export=ALL,RUN=$RUN,CONFIG=$CONFIG,STEPS=$STEPS,MODEL=$MODEL,MT=$MT,PORT_BASE=$((11000 + (SLURM_JOB_ID % 90) * 10)) \
    bash -c '
set -uo pipefail
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles/common.sh
{ git -C /root/miles log -1 --format="miles %H"; git -C /root/Megatron-LM log -1 --format="megatron %H"; git -C /sgl-workspace/sglang log -1 --format="sglang %H";
  python3 -c "import torch, sglang, fla, causal_conv1d, transformers; print(\"torch\", torch.__version__, \"sglang\", sglang.__version__, \"fla\", fla.__version__, \"causal_conv1d\", causal_conv1d.__version__, \"transformers\", transformers.__version__)"; } | tee $RUN/versions.txt
if grep -q "^--ref-load" $CONFIG; then convert_ckpt $MT $MODELS/$MODEL $CKPT/${MODEL}_torch_dist 4 || exit 1; fi
start_ray 8
echo "TRAIN_START $(date +%s)" >> $RUN/times.txt
ARGS=$(grep -v "^\s*#" $CONFIG | sed -e "s|\$RUN|$RUN|g" -e "s|\$STEPS|$STEPS|g" | tr "\n" " ")
python3 train_async.py $(model_args $MT) $ARGS ${EXTRA_ARGS:-} > $RUN/train.log 2>&1
rc=$?
echo "TRAIN_END $(date +%s) EXIT $rc" >> $RUN/times.txt
cp -r /tmp/ray/session_latest/logs $RUN/raylogs 2>/dev/null
stop_ray
exit $rc
'
rc=$?
echo "JOB_END $(date +%s) EXIT $rc" | tee -a $RUN/times.txt
exit $rc
