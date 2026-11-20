#!/bin/bash
# Prepare the harder-W2 candidate pool as local (.sif) Harbor tasks, then validate them on this CPU node with Harbor's
# singularity env: oracle (expect 1) and nop (expect 0).
#SBATCH -p hopper-cpu
#SBATCH --qos=normal
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J w2hard-prep
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl/w2-prep/%x-%j.out
set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
HERE=$BENCH/workloads/agentic/hard
V=/fsx/amine_dirhoussi/envs/bench-trl-w2
SIF=/fsx/amine_dirhoussi/stable-bench-src/container-tools/images/termgrade-base.sif
OUTD=/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/hardprobe
date -u
$V/bin/python -I $HERE/select_pool.py $HERE/candidates.txt || exit 1
$V/bin/python -I $BENCH/workloads/agentic/prepare.py --tasks $HERE/candidates.txt --out $OUTD --image $SIF || exit 1
date -u; echo PREPARED
[ "${VALIDATE:-1}" = 1 ] || exit 0
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh
BOX=/scratch/$USER/w2hard-$SLURM_JOB_ID; mkdir -p $BOX/tmp $BOX/c $BOX/t
export LD_LIBRARY_PATH=$AP_LIBS APPTAINER_BIND=/etc/resolv.conf TMPDIR=$BOX/tmp APPTAINER_CACHEDIR=$BOX/c APPTAINER_TMPDIR=$BOX/t
OUT=/fsx/amine_dirhoussi/stable-bench-runs/trl/w2-validate/hardprobe-$SLURM_JOB_ID; mkdir -p $OUT
for agent in nop oracle; do
    echo "== $agent $(date -u +%T)"
    $V/bin/harbor run -p $OUTD/tasks -a $agent -e singularity -n ${N:-12} -o "$OUT/$agent" --yes > "$OUT/$agent.log" 2>&1
    echo "harbor exit $?  $(date -u +%T)"
    $V/bin/python $BENCH/workloads/agentic/validate_summary.py "$OUT/$agent" --out "$OUT/$agent-summary.json"
done
pkill -u $USER -f "$BOX" 2>/dev/null; rm -rf $BOX
date -u
