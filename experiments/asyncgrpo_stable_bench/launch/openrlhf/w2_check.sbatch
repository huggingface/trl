#!/bin/bash
#SBATCH --job-name=orlhf-w2check
#SBATCH --partition=hopper-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/logs/%x-%j.out
# CPU check for the W2 agent glue: envs/openrlhf/w2_check.py against a real Harbor server on local Apptainer boxes.
set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
W2_VENV=/fsx/amine_dirhoussi/envs/bench-trl-w2
OPENENV=/fsx/amine_dirhoussi/stable-bench-src/openenv
SPLIT_DIR=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train
RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/openrlhf/w2check-$SLURM_JOB_ID
mkdir -p $RUN_DIR/episodes
SCR=/scratch/$USER; [ -d /scratch ] && [ -w /scratch ] || SCR=/tmp/$USER
BOX_DIR=$SCR/w2-boxes-$SLURM_JOB_ID; mkdir -p $BOX_DIR/tmp $BOX_DIR/apptainer-cache $BOX_DIR/apptainer-tmp
export OPENENV_HARBOR_TRIALS_DIR=$SCR/w2-trials-$SLURM_JOB_ID; mkdir -p $OPENENV_HARBOR_TRIALS_DIR
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh > $RUN_DIR/apptainer-env.log 2>&1
export W2_PYTHON=$W2_VENV/bin/python W2_PYTHONPATH=$OPENENV/envs:$BENCH/workloads/agentic W2_SANDBOX=singularity
export W2_EPISODES_DIR=$RUN_DIR/episodes AGENT_TIMEOUT=600 STEP_LIMIT=30
VENV=/fsx/amine_dirhoussi/envs/bench-openrlhf
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
MODEL=$(ls -d $HF_HUB_CACHE/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)
P=$((62000 + (SLURM_JOB_ID % 90) * 20))
$VENV/bin/python $BENCH/configs/openrlhf/w2_agent.py stub $P > $RUN_DIR/stub.log 2>&1 &
STUB=$!
until curl -sf http://127.0.0.1:$P/v1/models >/dev/null; do sleep 2; done
env LD_LIBRARY_PATH=$AP_LIBS APPTAINER_BIND=/etc/resolv.conf TMPDIR=$BOX_DIR/tmp APPTAINER_CACHEDIR=$BOX_DIR/apptainer-cache \
    APPTAINER_TMPDIR=$BOX_DIR/apptainer-tmp PATH=$W2_VENV/bin:$PATH PYTHONPATH=$W2_PYTHONPATH MAX_CONCURRENT_ENVS=16 \
    $W2_VENV/bin/python $BENCH/workloads/agentic/harbor_serve.py harbor serve --dataset $SPLIT_DIR --llm-url http://127.0.0.1:$P \
    --model policy --host 127.0.0.1 --port $((P + 2)) --capture-port $((P + 3)) --expose direct --max-output-tokens 8192 \
    --no-rollouts --no-run-history > $RUN_DIR/harbor.log 2>&1 &
HP=$!
until curl -sf http://localhost:$((P + 2))/health >/dev/null; do kill -0 $HP || { tail -30 $RUN_DIR/harbor.log; exit 1; }; sleep 5; done
kill $STUB
export HARBOR_URL=http://localhost:$((P + 2))
TR=$SCR/verl-trials-$SLURM_JOB_ID; mkdir -p $TR
tar xzf /fsx/amine_dirhoussi/stable-bench-runs/verl/verl-w2-M-shared-371895/trials.tgz -C $TR --wildcards '*/result.json' '*/agent/trajectory.json' 2>/dev/null
$VENV/bin/python $BENCH/envs/openrlhf/w2_check.py $MODEL $TR $SPLIT_DIR
RC=$?
kill $HP; sleep 3; pkill -u $USER -f $BOX_DIR; rm -rf $BOX_DIR $TR
exit $RC
