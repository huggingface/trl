#!/bin/bash
# End-of-job sandbox sweep for W2 jobs: cancel every running sandbox on the W2 image, unless another W2 job of mine
# (trl-w2*, skyrl-w2*, w2-validate*) is still running, since sandboxes carry no per-job tag. Then the last W2 job to end sweeps.
#   sweep.sh <python> <log>
others=$(squeue -u "$USER" -h -o '%i %j' | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && ($2 ~ /^(trl|skyrl)-w2/ || $2 ~ /^w2-validate/)' | wc -l)
HERE=$(dirname "$0")
if [ "$others" = 0 ]; then
    "$1" "$HERE/sweep_sandboxes.py" --cancel > "$2" 2>&1
else
    { echo "other W2 jobs running ($others): listing only"; "$1" "$HERE/sweep_sandboxes.py"; } > "$2" 2>&1
fi
