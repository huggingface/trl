#!/bin/bash -l
#SBATCH --job-name=miles-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
# Smoke: (A) HF->torch_dist for Qwen3.5-0.8B and 2B, (B) miles' smallest example = their Qwen3.5-0.8B CI recipe
# (tests/e2e/ckpt/test_qwen3_5_0.8B_ckpt.py, 2 GPUs colocated, train.py), (C) Qwen3.5-2B fully async on 2 train +
# 2 rollout GPUs with --save-hf, (D) compare exported HF keys with the original checkpoint.
set -uo pipefail
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.1.3.sqsh
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
OUT=/fsx/amine_dirhoussi/stable-bench-runs/miles/smoke-$SLURM_JOB_ID
mkdir -p $OUT
echo "JOB_START $(date +%s)"
# The image import kept getting preempted on hopper-cpu (low QOS), so it runs here when the image is missing.
[ -f $IMG ] || bash $BENCH/envs/miles/install.sh || exit 1
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 > $OUT/gpu.csv 2>/dev/null &
SMI=$!

# b200-queue submits from a -c 1 --mem=2G waiter with --export=ALL: its SLURM_MEM_PER_NODE / SLURM_CPUS_PER_TASK leak in
# and would cap this srun step at 2 GiB and 1 CPU. Drop them and give the step the whole allocation.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG --container-mounts=/fsx:/fsx --container-workdir=/root/miles \
    --export=ALL,OUT=$OUT,BENCH=$BENCH,PORT_BASE=$((11000 + (SLURM_JOB_ID % 90) * 10)) \
    bash -c '
set -uo pipefail
source $BENCH/launch/miles/common.sh
echo "== env"; nvidia-smi -L; python3 -c "import torch, sglang, transformers, fla, causal_conv1d, megatron.core as mc, transformer_engine as te; print(\"torch\", torch.__version__, torch.version.cuda, \"sglang\", sglang.__version__, \"transformers\", transformers.__version__, \"fla\", fla.__version__, \"causal_conv1d\", causal_conv1d.__version__, \"mcore\", mc.__version__, \"te\", te.__version__)"
git -C /root/miles log -1 --format="miles %H %s"; git -C /root/Megatron-LM log -1 --format="megatron %H"; git -C /sgl-workspace/sglang log -1 --format="sglang %H"

echo "== FA4 check (unpatched, then patched)"
PYTHONPATH=${PYTHONPATH//$BENCH\/configs\/miles\/fa4_window_fix:/} python3 $BENCH/launch/miles/fa4_check.py
python3 $BENCH/launch/miles/fa4_check.py
echo "== A convert"
convert_ckpt qwen3.5-0.8B $MODELS/Qwen3.5-0.8B $CKPT/Qwen3.5-0.8B_torch_dist 2 || exit 1
convert_ckpt qwen3.5-2B $MODELS/Qwen3.5-2B $CKPT/Qwen3.5-2B_torch_dist 2 || exit 1

echo "== B 0.8B CI recipe (colocated, 2 GPUs)"
export CUDA_VISIBLE_DEVICES_ALL=$CUDA_VISIBLE_DEVICES
start_ray 2
T0=$(date +%s)
python3 train.py $(model_args qwen3.5-0.8B) \
  --hf-checkpoint $MODELS/Qwen3.5-0.8B/ --ref-load $CKPT/Qwen3.5-0.8B_torch_dist \
  --prompt-data $DATA/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt --label-key label --apply-chat-template \
  --rollout-shuffle --rm-type deepscaler --num-rollout 2 --rollout-batch-size 4 --n-samples-per-prompt 2 \
  --rollout-max-response-len 1024 --rollout-temperature 0.8 --global-batch-size 8 --balance-data \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type k1 --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size \
  --max-tokens-per-gpu 16384 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-cuda-graph-bs-decode 1 2 4 8 16 \
  --check-weight-update-skip-list visual \
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend ${ATTN_BACKEND:-flash} --actor-num-nodes 1 --actor-num-gpus-per-node 2 --num-gpus-per-node 2 --colocate \
  > $OUT/B.log 2>&1
echo "B_EXIT $? WALL_S $(( $(date +%s) - T0 ))"
cp -r /tmp/ray/session_latest/logs $OUT/B-raylogs 2>/dev/null
stop_ray; sleep 5

echo "== C 2B fully async (2 train TP=2 + 2 rollout)"
start_ray 4
T0=$(date +%s)
python3 train_async.py $(model_args qwen3.5-2B) \
  --hf-checkpoint $MODELS/Qwen3.5-2B/ --ref-load $CKPT/Qwen3.5-2B_torch_dist \
  --save $OUT/C-megatron --save-interval 2 --save-hf $OUT/C-hf/step_{rollout_id} \
  --fully-async --pause-generation-mode in_place --use-tis \
  --prompt-data /fsx/amine_dirhoussi/stable-bench-runs/miles/data/w1_train.parquet --input-key messages --label-key label \
  --apply-chat-template --custom-rm-path w1_reward.reward --rollout-max-prompt-len 1024 --seed 42 \
  --calculate-per-token-loss --disable-grpo-std-normalization \
  --num-rollout 4 --rollout-batch-size 8 --n-samples-per-prompt 4 \
  --rollout-max-response-len 2048 --rollout-temperature 1 --global-batch-size 32 --balance-data \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.999 \
  --advantage-estimator grpo --kl-loss-coef 0.00 --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size \
  --max-tokens-per-gpu 9216 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --check-weight-update-skip-list visual \
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend ${ATTN_BACKEND:-flash} --actor-num-nodes 1 --actor-num-gpus-per-node 2 --num-gpus-per-node 4 --rollout-num-gpus 2 \
  > $OUT/C.log 2>&1
echo "C_EXIT $? WALL_S $(( $(date +%s) - T0 ))"
cp -r /tmp/ray/session_latest/logs $OUT/C-raylogs 2>/dev/null
stop_ray

echo "== D exported HF keys vs original"
python3 - <<EOF
import json, glob, os
from safetensors import safe_open
def keys(d):
    ks = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys(): ks[k] = tuple(s.get_slice(k).get_shape())
    return ks
orig = keys("$MODELS/Qwen3.5-2B")
for d in sorted(glob.glob("$OUT/C-hf/step_*")):
    ex = keys(d)
    miss = sorted(set(orig) - set(ex)); extra = sorted(set(ex) - set(orig))
    bad = [k for k in set(orig) & set(ex) if orig[k] != ex[k]]
    print(d, "exported", len(ex), "orig", len(orig), "missing", len(miss), "extra", len(extra), "shape_mismatch", len(bad))
    print("  missing prefixes:", sorted({".".join(k.split(".")[:3]) for k in miss})[:10])
    print("  extra:", extra[:10]); print("  files:", sorted(os.listdir(d)))
EOF
echo "== D2 complete the export with the frozen base tensors, then load it with transformers"
for d in $OUT/C-hf/step_*; do python3 $BENCH/configs/miles/complete_hf_export.py $MODELS/Qwen3.5-2B $d; done
d=$(ls -d $OUT/C-hf/step_* | tail -1)
python3 -c "
import sys, torch
from transformers import AutoModelForImageTextToText
m, info = AutoModelForImageTextToText.from_pretrained(sys.argv[1], dtype=torch.bfloat16, output_loading_info=True)
print(type(m).__name__, {k: len(v) for k, v in info.items()}, {k: sorted(v)[:5] for k, v in info.items() if v})
" "$d"
'
echo "SRUN_EXIT $?"
kill $SMI 2>/dev/null
echo "JOB_END $(date +%s)"
