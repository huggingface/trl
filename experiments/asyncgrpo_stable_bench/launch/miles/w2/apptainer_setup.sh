# Sourced inside the miles container by launch/miles/w2.sbatch (read at run time, so it can change while a job queues).
# Makes Harbor's `singularity` (Apptainer 1.5.5, unprivileged) work nested in the Pyxis container on a GPU node, then
# picks the first box variant that starts, logging which. Needs: W2, RUN, BOX_DIR.
# 1. libs the Ubuntu 24.04 image lacks, copied into the image lib dir (the starter drops LD_LIBRARY_PATH).
L=$W2/apptainer-libs-24.04
cp -a $L/usr/lib/x86_64-linux-gnu/. $L/lib/x86_64-linux-gnu/. /usr/lib/x86_64-linux-gnu/ && ldconfig
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh > $RUN/apptainer-env.log 2>&1
# 2. the image has /usr/bin/fakeroot; its libfakeroot (glibc 2.38) breaks the Debian 12 box.
export APPTAINER_IGNORE_FAKEROOT_COMMAND=1
export TMPDIR=$BOX_DIR/tmp APPTAINER_CACHEDIR=$BOX_DIR/apptainer-cache APPTAINER_TMPDIR=$BOX_DIR/apptainer-tmp
mkdir -p $TMPDIR $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
# 3. Harbor runs boxes with --containall (new PID namespace, so Apptainer mounts a fresh /proc). On a GPU node the
#    container's /proc carries the NVIDIA hook's overmounts, and the kernel refuses a new proc mount in a nested user
#    namespace unless proc is fully visible (job 372145: "can't mount proc filesystem to /proc: operation not permitted").
#    Variants, first that works wins: (a) as is; (b) a private copy of the install with `mount proc = no` plus the
#    container's /proc bound in; (c) same copy, no /proc in the box.
NOPROC=$W2/apptainer-1.5.5-noproc
if [ ! -x $NOPROC/usr/bin/apptainer ]; then
    rm -rf $NOPROC.tmp && cp -a $AP $NOPROC.tmp && sed -i "s/^mount proc = yes/mount proc = no/" $NOPROC.tmp/etc/apptainer/apptainer.conf \
        && rm -rf $NOPROC.tmp/bin-shim && mkdir $NOPROC.tmp/bin-shim && mv $NOPROC.tmp $NOPROC
    ln -sfn $NOPROC/usr/bin/apptainer $NOPROC/bin-shim/singularity; ln -sfn $NOPROC/usr/bin/apptainer $NOPROC/bin-shim/apptainer
fi
SIF=/fsx/amine_dirhoussi/stable-bench-src/container-tools/images/termgrade-base.sif
BASE_PATH=$PATH
_box_try() {  # <name> <bin-shim dir> <APPTAINER_BIND>
    PATH=$2:$BASE_PATH APPTAINER_BIND=$3 timeout 120 singularity exec --no-mount home,tmp --pwd /app --writable-tmpfs \
        --fakeroot --containall --pid $SIF bash -c "id -u; ls /proc | wc -l; tmux -V; python3 -c \"import socket; print(socket.gethostbyname(\\\"pypi.org\\\"))\"" \
        > $RUN/box-variant-$1.log 2>&1
}
APPTAINER_VARIANT=""
for v in "a $AP/bin-shim /etc/resolv.conf" "b $NOPROC/bin-shim /etc/resolv.conf,/proc" "c $NOPROC/bin-shim /etc/resolv.conf"; do
    set -- $v
    if _box_try $1 $2 $3; then APPTAINER_VARIANT=$1; export PATH=$2:$BASE_PATH APPTAINER_BIND=$3; break; fi
done
echo "APPTAINER_VARIANT ${APPTAINER_VARIANT:-none} PATH_SHIM $(dirname $(which singularity)) APPTAINER_BIND ${APPTAINER_BIND:-}" | tee -a $RUN/versions.txt
for f in $RUN/box-variant-*.log; do echo "== $f"; tail -3 $f; done
[ -n "$APPTAINER_VARIANT" ]
