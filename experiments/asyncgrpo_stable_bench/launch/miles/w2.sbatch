#!/bin/bash -l
#SBATCH --job-name=miles-w2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
# W2 agentic (TermGrade + terminus-2) on miles, one node: 4 SGLang engines (TP=1) + 4 Megatron GPUs (TP=4), fully async.
# Harbor runs in-process in the rollout workers (examples/experimental/harbor) with HARBOR_ENV_TYPE=singularity: every
# episode's box is an unprivileged Apptainer container started from inside miles' Pyxis container (nested).
# Submit: PRIO_TAG=PRIO_TRAIN b200-queue launch/miles/w2.sbatch   (env TAG=shared, STEPS=60, SAVE_STEPS=20, EXTRA_ARGS)
# MOE=1: Qwen3.5-35B-A3B, LoRA via Megatron-Bridge (configs/miles/w2-moe.args, no torch_dist conversion).
# Config: configs/miles/w2-M.args; knob mapping in configs/miles/EQUIVALENCE.md (W2).
set -uo pipefail
if [ "${MOE:-0}" = 1 ]; then SIZE=moe; MODEL=Qwen3.5-35B-A3B; MT=qwen3.5-35B-A3B_lora
else SIZE=M; MODEL=Qwen3.5-9B; MT=qwen3.5-9B; fi
TAG=${TAG:-shared}; STEPS=${STEPS:-60}; SAVE_STEPS=${SAVE_STEPS:-20}
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.1.3.sqsh
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
CONFIG=$BENCH/configs/miles/w2-$SIZE.args
RUN_ID=miles-w2-$SIZE-$TAG-$SLURM_JOB_ID
RUN=/fsx/amine_dirhoussi/stable-bench-runs/miles/$RUN_ID
BOX_DIR=/scratch/$USER/w2-boxes-$SLURM_JOB_ID
mkdir -p $RUN $BOX_DIR/tmp
cp $CONFIG $RUN/config.args
echo "RUN_ID $RUN_ID  RUN $RUN  NODE $(hostname)  $(date -u)"
echo "JOB_START $(date +%s)" | tee $RUN/times.txt

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 \
    | awk -v n=$(hostname) 'NR==1{print $0", node"; fflush(); next}{print $0", "n; fflush()}' > $RUN/gpu.csv &
SMI=$!
python3 $BENCH/launch/miles/w2/box_usage.py $BOX_DIR 5 > $RUN/boxes.csv 2>/dev/null &
BOXMON=$!
cleanup_boxes() {
    # Boxes are processes of this job (Enroot shares the host PID namespace): stop any left, then remove scratch.
    local n=$(pgrep -u $USER -fc "$BOX_DIR")
    pkill -u $USER -f "$BOX_DIR" 2>/dev/null; sleep 5; pkill -9 -u $USER -f "$BOX_DIR" 2>/dev/null
    echo "box processes at cleanup: $n; left after: $(pgrep -u $USER -fc "$BOX_DIR") ($(date -u))" | tee -a $RUN/sweep.log
    rm -rf $BOX_DIR
}
trap 'kill $SMI $BOXMON 2>/dev/null; cleanup_boxes' EXIT

unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG \
    --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/root/miles \
    --export=ALL,RUN=$RUN,CONFIG=$CONFIG,STEPS=$STEPS,SAVE_STEPS=$SAVE_STEPS,MODEL=$MODEL,MT=$MT,BOX_DIR=$BOX_DIR,PORT_BASE=$((11000 + (SLURM_JOB_ID % 90) * 10)) \
    bash -c '
set -uo pipefail
source /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles/common.sh
W2=/fsx/amine_dirhoussi/stable-bench-runs/miles/w2
# Harbor at miles pinned branch (harbor-miles-v0.20.0 head, pinned by sha), installed into the container env where the
# rollout workers import it (examples/experimental/harbor/README.md step 1). uv resolves against the image packages.
t0=$(date +%s)
UV_CACHE_DIR=$W2/uv-cache $W2/bin/uv pip install --python $(which python3) \
    "harbor @ git+https://github.com/harbor-framework/harbor@c699a7ec3bd9d8d07d779ff7f90c722a5d7a0c22" > $RUN/harbor_install.log 2>&1 \
    || { echo "harbor install failed"; tail -20 $RUN/harbor_install.log; exit 1; }
echo "HARBOR_INSTALL_S $(( $(date +%s) - t0 ))"
{ git -C /root/miles log -1 --format="miles %H"; git -C /root/Megatron-LM log -1 --format="megatron %H"; git -C /sgl-workspace/sglang log -1 --format="sglang %H";
  python3 -c "import torch, sglang, fla, transformers, importlib.metadata as m; print(\"torch\", torch.__version__, \"sglang\", sglang.__version__, \"fla\", fla.__version__, \"transformers\", transformers.__version__, \"harbor\", m.version(\"harbor\"), \"litellm\", m.version(\"litellm\"))";
  echo "harbor git c699a7ec3bd9d8d07d779ff7f90c722a5d7a0c22 (harbor-miles-v0.20.0)"; } | tee $RUN/versions.txt

# Apptainer for the nested boxes (libs, fakeroot, /proc variants): launch/miles/w2/apptainer_setup.sh, read at run time.
source $BENCH/launch/miles/w2/apptainer_setup.sh || { echo "no Apptainer box variant starts"; exit 1; }

# Harbor-side knobs (read by harbor_agent_function.py in the rollout workers; Ray workers inherit this env).
export HARBOR_ENV_TYPE=singularity HARBOR_ENV_KWARGS="{\"singularity_no_mount\":\"home,tmp\"}"
export HARBOR_TASKS_DIR=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train/tasks
export HARBOR_TRIALS_DIR=$RUN/trials AGENT_MODEL_NAME=model
export AGENT_TIMEOUT=600 AGENT_MAX_OUTPUT_TOKENS=8192 AGENT_MAX_INPUT_TOKENS=65536 HARBOR_MAX_SEQ_LEN=65536
export HARBOR_AGENT_MAX_ITERATIONS=30 HARBOR_TERMINUS_2_ENABLE_SUMMARIZE=false HARBOR_TIMEOUT_MULTIPLIER=1.0
export PYTHONPATH=$PYTHONPATH:/root/miles/examples/experimental/harbor:/root/miles/examples/swe-agent-harbor-docker
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
env | grep -E "^(HARBOR|AGENT|APPTAINER)" | sort > $RUN/harbor_env.txt

# Data: one row per train task (prompt = instruction; Harbor reads the task dir by metadata.instance_id).
DATA=$W2/data; mkdir -p $DATA
python3 - <<PY
import json, pathlib
tasks = pathlib.Path("$HARBOR_TASKS_DIR")
names = [l.strip() for l in open("$BENCH/workloads/agentic/train_tasks.txt") if l.strip()]
with open("$DATA/w2_train.jsonl", "w") as f:
    for n in names:
        f.write(json.dumps({"prompt": (tasks / n / "instruction.md").read_text(), "metadata": {"instance_id": n, "agent_name": "terminus-2"}}) + "\n")
print("w2_train rows", len(names))
PY

if grep -q "^--ref-load" $CONFIG; then convert_ckpt $MT $MODELS/$MODEL $CKPT/${MODEL}_torch_dist 4 || exit 1; fi
start_ray 8
echo "TRAIN_START $(date +%s)" >> $RUN/times.txt
ARGS=$(grep -v "^\s*#" $CONFIG | sed -e "s|\$RUN|$RUN|g" -e "s|\$STEPS|$STEPS|g" -e "s|\$DATA|$DATA|g" -e "s|^--save-interval 20$|--save-interval $SAVE_STEPS|" | tr "\n" " ")
python3 train_async.py $(model_args $MT) $ARGS ${EXTRA_ARGS:-} > $RUN/train.log 2>&1
rc=$?
echo "TRAIN_END $(date +%s) EXIT $rc" >> $RUN/times.txt
cp -r /tmp/ray/session_latest/logs $RUN/raylogs 2>/dev/null
stop_ray
for d in $RUN/hf/step_*; do [ -d "$d" ] && python3 $BENCH/configs/miles/complete_hf_export.py $MODELS/$MODEL $d; done
for d in $RUN/hf/step_*; do [ -d "$d" ] || continue; r=${d##*_}; ln -sfn hf/step_$r $RUN/checkpoint-$((r + 1)); done
exit $rc
' &
SRUN=$!
# In-job smoke checks (launch/miles/w2/watch.py): on the first failed check, stop the srun step with a clear message.
( python3 $BENCH/launch/miles/w2/watch.py $RUN $SAVE_STEPS > $RUN/watch.log 2>&1
  if [ $? = 3 ]; then echo "WATCH_KILL: $(grep WATCH_KILL $RUN/watch.log | tail -1) ($(date -u)); stopping the run" | tee -a $RUN/times.txt; kill $SRUN; fi ) &
WATCH=$!
wait $SRUN
rc=$?
kill $WATCH 2>/dev/null
echo "JOB_END $(date +%s) EXIT $rc" | tee -a $RUN/times.txt
exit $rc
