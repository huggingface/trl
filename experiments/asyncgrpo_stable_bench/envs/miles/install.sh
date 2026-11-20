#!/bin/bash
# miles install, from scratch. Recommended path upstream is the Docker image (docs/getting-started/installation.md),
# which on this cluster means importing it with Enroot into a squashfs that Pyxis runs.
# Run inside a Slurm job (enroot needs node-local /scratch, ~60 GB of it);
#   it runs at the top of launch/miles/smoke.sbatch when the image is missing (hopper-cpu low-QOS imports were preempted twice).
set -euo pipefail
TAG=${MILES_TAG:-v0.1.3}
SRC=/fsx/amine_dirhoussi/stable-bench-src/miles
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-${TAG}.sqsh
T0=$(date +%s)

# 1. source checkout (for reading code and launch scripts; the image carries its own copy at /root/miles)
[ -d "$SRC" ] || git clone https://github.com/radixark/miles.git "$SRC"
git -C "$SRC" fetch --tags -q && git -C "$SRC" checkout -q ${MILES_SHA:-ace18d49e9c942bfa45451b12ac3bbbc7486969a}  # = /root/miles in the v0.1.3 image (release/v0.1.3 head, tag + 1 hotfix)

# 2. container image -> squashfs on /fsx
# write to a temp name and rename on success: a preempted import otherwise leaves a truncated .sqsh behind
rm -f "$IMG" "$IMG.part"
enroot import -o "$IMG.part" "docker://radixark/miles:${TAG}"
mv "$IMG.part" "$IMG"
T1=$(date +%s)
echo "IMPORT_WALL_S=$((T1-T0))"
ls -l "$IMG"
