#!/bin/bash
#SBATCH --job-name=miles-w2-probe3
#SBATCH --partition=hopper-cpu
#SBATCH --qos=normal
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
# Nested Apptainer inside the miles container, take 2: the container (Ubuntu 24.04) lacks four libs the Apptainer 1.5.5
# binaries link (libsubid4, libfuse3-3, liblzo2-2, libprotobuf-c1). Fetch the noble .debs on this 24.04 host (apt-get
# download, no root) into a miles-only dir and put it on LD_LIBRARY_PATH inside the container.
set -uo pipefail
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.1.3.sqsh
W2=/fsx/amine_dirhoussi/stable-bench-runs/miles/w2
W=$W2/probe-$SLURM_JOB_ID; mkdir -p $W
L=$W2/apptainer-libs-24.04
if [ ! -e $L/.done ]; then
  rm -rf $L; mkdir -p $L/debs
  ( cd $L/debs && apt-get download libsubid4 libfuse3-3 liblzo2-2 libprotobuf-c1 libtalloc2 2>&1 )
  for d in $L/debs/*.deb; do dpkg-deb -x $d $L; done
  ls $L/usr/lib/x86_64-linux-gnu $L/lib/x86_64-linux-gnu 2>/dev/null
  touch $L/.done
fi
echo "host $(hostname) $(date -u)"
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/root/miles \
  --export=ALL,W=$W,W2=$W2,L=$L bash -c '
UV_CACHE_DIR=$W2/uv-cache $W2/bin/uv pip install --python $(which python3) "harbor @ git+https://github.com/harbor-framework/harbor@c699a7ec3bd9d8d07d779ff7f90c722a5d7a0c22" 2>&1 | tail -3
# Apptainer starter re-execs itself with a clean environment (no LD_LIBRARY_PATH), so the libs go into the container's
# own lib dir (the Enroot rootfs is writable and owned by us), then ldconfig.
cp -a $L/usr/lib/x86_64-linux-gnu/. $L/lib/x86_64-linux-gnu/. /usr/lib/x86_64-linux-gnu/ && ldconfig 2>&1 | tail -2; ldconfig -p | grep -E "libsubid|libfuse3|liblzo2|libprotobuf-c"
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh
for b in $AP/usr/bin/apptainer $AP/usr/libexec/apptainer/bin/*; do ldd $b 2>/dev/null | grep "not found" | sed "s|^|$b |"; done
apptainer --version
BOX=/scratch/$USER/w2probe-$SLURM_JOB_ID; mkdir -p $BOX/tmp
export TMPDIR=$BOX/tmp APPTAINER_CACHEDIR=$BOX/cache APPTAINER_TMPDIR=$BOX/atmp APPTAINER_BIND=/etc/resolv.conf
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
SIF=/fsx/amine_dirhoussi/stable-bench-src/container-tools/images/termgrade-base.sif
time apptainer exec --fakeroot --containall --writable-tmpfs --no-mount home,tmp $SIF bash -c "id; head -2 /etc/os-release; python3 --version; getent hosts pypi.org"
echo "NESTED_EXEC_RC $?"
export HARBOR_ENV_KWARGS="{\"singularity_no_mount\":\"home,tmp\"}"
time python3 /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles/w2/oracle_trials.py \
   /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train/tasks $W/trials 8
echo ORACLE_RC $?
echo "boxes left: $(pgrep -u $USER -fc $BOX)"; pkill -9 -u $USER -f $BOX; rm -rf $BOX
'
echo "END $(date -u)"
