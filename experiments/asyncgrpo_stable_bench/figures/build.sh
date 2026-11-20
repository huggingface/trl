#!/bin/bash
# Regenerate every figure: fair throughput table + text-to-SQL 35B tables -> figures/data/bench.json -> static blog figures (figures/post/),
# diagnostic charts (figures/out/) -> interactive page (figures/interactive/index.html).
#   figures/build.sh          submits itself to hopper-cpu and waits (never runs the work on the login node)
# Runs that are still in flight show as "in progress"; rerun this once their files land.
set -euo pipefail
# BENCH: this directory. PY: any Python with numpy, pandas and matplotlib. RUNS: native run outputs (read by
# collect/t2sql_35b.py and parts of figures/build_data.py; everything else comes from results/raw/).
BENCH=${BENCH:-/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench}
PY=${PY:-/fsx/amine_dirhoussi/envs/bench-trl/bin/python}
export RUNS=${RUNS:-/fsx/amine_dirhoussi/stable-bench-runs}
LOGS=${LOGS:-$RUNS/figures-logs}
# Built-in-options runs only (SHARED_W1 rev. 8): patched/custom-loss runs, recommended arms and smokes stay out.
RUNS='^(trl-w1-S-shared-builtin-|trl-w1-M-shared-|prime-rl-w1-S-shared-builtin-|prime-rl-w1-M-shared-|openrlhf-w1-[SM]-shared-|skyrl-w1-[SM]-shared-|verl-w1-S-shared-|trl-w2-M-local-|trl-w2-M-shared-|prime-rl-w2-M-shared-[0-9])'

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$LOGS"
    exec sbatch --wait -p hopper-cpu --qos=low -c 4 --mem=16G -t 1:00:00 --no-requeue -J figures \
        -o "$LOGS/%x-%j.out" "$0"
fi

cd "$BENCH"
"$PY" -c "import matplotlib" 2>/dev/null || uv pip install --python "$PY" matplotlib
"$PY" collect/throughput_fair.py
"$PY" collect/t2sql_35b.py
"$PY" figures/build_data.py
"$PY" figures/charts.py --diagnostics --runs "$RUNS"
"$PY" figures/build_interactive.py
echo "done: figures/post/, figures/out/, figures/interactive/index.html"
