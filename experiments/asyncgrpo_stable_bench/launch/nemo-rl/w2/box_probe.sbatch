#!/bin/bash
# Which Apptainer flags start a TermGrade box inside the NeMo-RL Pyxis container on a hopper-b200 node.
#SBATCH -J nemo-w2-boxprobe
#SBATCH --nodes=1
#SBATCH -t 0:40:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --mem=0 --container-image=$IMG --container-mounts=/fsx:/fsx,/scratch:/scratch bash -c '
export APPTAINER_IGNORE_FAKEROOT_COMMAND=1 APPTAINER_CACHEDIR=/tmp/apc APPTAINER_TMPDIR=/tmp/apt
SIF=/fsx/amine_dirhoussi/stable-bench-src/container-tools/images/termgrade-base.sif
B="--no-mount home --no-mount tmp --no-mount bind-paths --writable-tmpfs --fakeroot"
cat /proc/self/mountinfo | grep -c " /proc/" ; uname -r
t() { echo "== $1"; shift; timeout 120 singularity exec "$@" $SIF bash -c "ls /proc | head -3; ps aux | wc -l" 2>&1 | tail -4; }
t gym_flags $B --containall --pid
t no_pid_contain $B --contain --cleanenv --ipc
t containall_noproc_bindproc $B --containall --pid --no-mount proc -B /proc
t contain_noproc_bindproc $B --contain --cleanenv --ipc --no-mount proc -B /proc
echo; echo host:; unshare -Urpf --mount-proc true && echo unshare_proc_ok
'
