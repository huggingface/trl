#!/bin/bash
# Text-to-SQL (miles' River recipe, TRL's port) on Qwen3.5-35B-A3B + LoRA with prime-rl: 1 node, 4 inference + 4 trainer
# GPUs, 100 steps (configs/prime-rl/t2sql-moe.toml, env configs/prime-rl/t2sql_env.py).
#   PRIO_TAG=PRIO_TRAIN b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/prime-rl/w2_t2sql.sbatch
# The PEFT adapters of KEEP_ADAPTERS (default 20 40 60 80 100) are copied out of prime-rl's rolling broadcasts/ dir as
# they land and merged at once into hf/step_N (envs/prime-rl/merge_lora.py, on this node's CPUs), which then gets a
# merge_done.json; the hopper-cpu helper envs/prime-rl/t2sql_evals.sh queues each eval from there. An in-job watcher
# (envs/prime-rl/t2sql_watch.py) cancels the job if the first steps are broken.
# Env knobs: CONFIG (path under $BENCH, default configs/prime-rl/t2sql-moe.toml), TAG (appended to the run_id), MAX_STEPS (-> --max-steps), KEEP_ADAPTERS.
#SBATCH --job-name=prl-t2sql
#SBATCH --partition=hopper-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out

set -uo pipefail
SRC=/fsx/amine_dirhoussi/stable-bench-src/prime-rl
VENV=/fsx/amine_dirhoussi/envs/bench-prime-rl
D=/fsx/amine_dirhoussi/stable-bench-src/text2sql
export BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
CFG=$BENCH/${CONFIG:-configs/prime-rl/t2sql-moe.toml}
RUNS=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl
RUN_ID=prime-rl-t2sql-Qwen3.5-35B-A3B-lora32${TAG:+-$TAG}-$SLURM_JOB_ID
OUT=$RUNS/$RUN_ID
AUX=$OUT-aux # sidecar files: rl refuses to start in a non-empty run dir
KEEP_ADAPTERS=${KEEP_ADAPTERS:-20 40 60 80 100}
mkdir -p "$AUX/bin"
EXTRA=()
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")

source "$VENV/bin/activate"
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub PYTHONUNBUFFERED=1
# the env (t2sql_env.py) and River's recipe modules (stdlib-only, as TRL's run)
export PYTHONPATH=$D/river-recipes/text-to-sql/src:$BENCH/configs/prime-rl
# SQLite snapshots and the recipe's temp files on node-local NVMe, as TRL's run
S=/scratch/$USER/prl-t2sql-$SLURM_JOB_ID; mkdir -p -m 700 $S $S/snapshots $S/tmp $S/home; chmod 700 $S/snapshots
export TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT=$S/snapshots TMPDIR=$S/tmp
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))

# rl starts each env worker pool as `env-worker-pool @ <cfg>` from PATH. verifiers' subprocess runtime makes a workdir per
# rollout under ~/.cache/verifiers (/admin, NFS); this wrapper points HOME at node-local disk for the env workers only.
cat > "$AUX/bin/env-worker-pool" <<EOF
#!/bin/bash
HOME=$S/home exec "$VENV/bin/env-worker-pool" "\$@"
EOF
chmod +x "$AUX/bin/env-worker-pool"
export PATH=$AUX/bin:$PATH

cleanup() {
    trap - EXIT TERM INT
    kill $SMI $KEEPER $WATCH 2>/dev/null
    wait $KEEPER 2>/dev/null
    rm -rf "$S"
    echo "job_end $(date +%s.%N)" | tee -a "$AUX/job_times.txt"
}
SMI=; KEEPER=; WATCH=
trap cleanup EXIT
trap 'exit 143' TERM INT

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 5 \
    | sed -u "1s/\$/, node/; 2,\$s/\$/, $(hostname)/" > "$AUX/gpu.csv" &
SMI=$!

# prime-rl keeps only the last two broadcasts/step_N: copy each wanted adapter once its .finished marker is written, then
# merge it into a full HF dir (one at a time, in this background loop).
( todo="$KEEP_ADAPTERS"; while [ -n "$todo" ]; do left=""; for n in $todo; do
    d=$OUT/broadcasts/step_$n
    if [ -f "$d/.finished" ]; then
        cp -r "$d" "$AUX/adapter_step_$n" && echo "adapter_kept $(date +%s.%N) step=$n" >> "$AUX/job_times.txt"
        echo "merge_start $(date +%s.%N) step=$n" >> "$AUX/job_times.txt"
        nice python "$BENCH/envs/prime-rl/merge_lora.py" "$AUX/adapter_step_$n" "$OUT/hf/step_$n" > "$AUX/merge-$n.json" 2> "$AUX/merge-$n.err" \
            && mv "$AUX/merge-$n.json" "$OUT/hf/step_$n/merge_done.json"
        echo "merge_end $(date +%s.%N) step=$n rc=$?" >> "$AUX/job_times.txt"
    else left="$left $n"; fi; done; todo=${left# }; sleep 10; done ) &
KEEPER=$!

python "$BENCH/envs/prime-rl/t2sql_watch.py" "$OUT" "$AUX" "$SLURM_JOB_ID" &
WATCH=$!

echo "job_start $(date +%s.%N) run_id=$RUN_ID node=$(hostname) port=$PORT cfg=$CFG" | tee "$AUX/job_times.txt"
cd "$SRC"
rl @ "$CFG" \
    --output-dir "$RUNS" \
    --run.name "$RUN_ID" \
    --inference.server.port "$PORT" \
    --orchestrator.model.client.base-url "http://localhost:$PORT/v1" \
    --rollout-transport.type zmq --rollout-transport.port $((PORT + 3)) \
    --monitors.wandb.project stable-bench-prime-rl --monitors.wandb.name "$RUN_ID" \
    --no-monitors.prime \
    "${EXTRA[@]}"
rc=$?
echo "train_end $(date +%s.%N) rc=$rc" | tee -a "$AUX/job_times.txt"
# let the last adapter's copy + merge finish (the keeper exits once every kept step is merged)
[ $rc = 0 ] && timeout 1200 bash -c "while kill -0 $KEEPER 2>/dev/null; do sleep 10; done"
exit $rc
