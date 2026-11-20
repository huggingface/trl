#!/bin/bash
#SBATCH -J nemo-rl-import
#SBATCH -p hopper-cpu
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
# NeMo-RL install as recommended (docs/about/installation.md: the NGC container). Enroot import to node-local
# /scratch, then copy the squashfs to /fsx under a temp name and rename on success.
set -euo pipefail
TAG=${NEMO_RL_TAG:-v0.8.0}
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-${TAG}.sqsh
[ -f $IMG ] && { echo "have $IMG"; exit 0; }
df -h /scratch
W=/scratch/$USER/enroot-$SLURM_JOB_ID; mkdir -p $W/data $W/tmp $W/cache
export ENROOT_CACHE_PATH=$W/cache ENROOT_DATA_PATH=$W/data ENROOT_TEMP_PATH=$W/tmp TMPDIR=$W/tmp ENROOT_MAX_PROCESSORS=8
T0=$(date +%s); echo "start $(date -u +%FT%TZ) $(hostname)"
enroot import -o $W/img.sqsh "docker://nvcr.io#nvidia/nemo-rl:${TAG}"
echo "squashed $(date -u +%FT%TZ) wall $(( $(date +%s) - T0 )) s"; ls -l $W/img.sqsh
cp $W/img.sqsh $IMG.part && mv $IMG.part $IMG
echo "done $(date -u +%FT%TZ) wall $(( $(date +%s) - T0 )) s"; ls -l $IMG
rm -rf $W
