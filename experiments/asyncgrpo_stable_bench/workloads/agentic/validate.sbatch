#!/bin/bash
# Validate a prepared TermGrade split on HF Sandboxes with Harbor's own agents: oracle (reference solution, expect 1)
# and nop (nothing, expect 0). Exercises the exact path training uses (image pull, environment/ upload, healthcheck
# setup, verifier upload). CPU only.
#   SPLIT=probe sbatch workloads/agentic/validate.sbatch
#SBATCH -p hopper-cpu
#SBATCH --qos=low
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 3:00:00
#SBATCH -J w2-validate
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl/w2-prep/%x-%j.out
set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
V=/fsx/amine_dirhoussi/envs/bench-trl-w2
SPLIT=${SPLIT:-probe}; N=${N:-16}; AGENTS=${AGENTS:-oracle nop}
OUT=/fsx/amine_dirhoussi/stable-bench-runs/trl/w2-validate/$SPLIT-$SLURM_JOB_ID
mkdir -p "$OUT"
trap '$BENCH/workloads/agentic/sweep.sh $V/bin/python "$OUT/sweep.log"' EXIT
for agent in $AGENTS; do
    echo "== $agent $(date -u +%T)"
    $V/bin/harbor run -p /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor/$SPLIT/tasks -a $agent -e hf-sandbox \
        -n $N -o "$OUT/$agent" --yes > "$OUT/$agent.log" 2>&1
    echo "harbor exit $?  $(date -u +%T)"
    $V/bin/python $BENCH/workloads/agentic/validate_summary.py "$OUT/$agent" --out "$OUT/$agent-summary.json"
done
