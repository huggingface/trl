#!/bin/bash
# Inside the NeMo-RL container: NeMo-RL's NeMo Gym GRPO runner + the watcher.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
cd /opt/nemo-rl
# the container venv, not the login uv/PATH leaking in through --export=ALL
export PATH=/opt/nemo_rl_venv/bin:$PATH
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
{ git log -1 --format='nemo-rl %h %cd' 2>/dev/null; git -C 3rdparty/Gym-workspace/Gym log -1 --format='gym %h %cd' 2>/dev/null;
  apptainer --version; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; } | tee -a $RUN_DIR/times.txt
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
# short Ray session path (AF_UNIX 107-byte limit)
export TMPDIR=/tmp/n$SLURM_JOB_ID RAY_TMPDIR=/tmp/n$SLURM_JOB_ID
mkdir -p $TMPDIR
# local boxes: Apptainer's cache/tmp on node-local disk; resolv.conf bound explicitly (Apptainer >= 1.5 drops it under
# Harbor's --no-mount; same as launch/trl/w2.sbatch)
mkdir -p $WORK/apptainer-cache $WORK/apptainer-tmp $WORK/harbor-jobs $WORK/sif-cache
export APPTAINER_CACHEDIR=$WORK/apptainer-cache APPTAINER_TMPDIR=$WORK/apptainer-tmp APPTAINER_BIND=/etc/resolv.conf,/proc
# Gym's SingularityEnvironment always passes --fakeroot; the container's libfakeroot needs GLIBC 2.38, newer than the
# box image, so use the root-mapped user namespace alone (Apptainer's own --ignore-fakeroot-command, as an env var)
export APPTAINER_IGNORE_FAKEROOT_COMMAND=1
# Gym drops pathspec from every server venv (copyleft exclusion in its pyproject); Harbor imports it via dirhash/scantree
export UV_OVERRIDE=$HERE/uv-override.txt
CONFIG=${CONFIG:-$BENCH/configs/nemo-rl/w2-M.yaml}
cp $CONFIG $RUN_DIR/config.yaml
echo "$(date -u +%FT%TZ) train" | tee -a $RUN_DIR/times.txt
setsid python examples/nemo_gym/run_grpo_nemo_gym.py --config $CONFIG ${OVERRIDES:-} >> $RUN_DIR/train.log 2>&1 &
PID=$!
python $HERE/watch.py $RUN_DIR $PID > $RUN_DIR/watch.out 2>&1 &
( while sleep 60; do echo "$(date -u +%FT%TZ) pty_nr=$(cat /proc/sys/kernel/pty/nr) pty_max=$(cat /proc/sys/kernel/pty/max) pts=$(ls /dev/pts | wc -l) singularity=$(pgrep -fc 'singularity exec') tmux=$(pgrep -fc tmux)" >> $RUN_DIR/pty.log
  # Reaper (harness glue): tmux servers of finished boxes outlive them and each holds PTYs (398039 ran out of PTYs
  # after ~2,600 episodes). A box lives <= ~16 min (setup 60 s + agent 600 s + verifier 300 s), so a tmux older
  # than 30 min belongs to a box that is gone.
  ps -u $USER -o pid=,etimes=,comm= | awk '$3 ~ /^tmux/ && $2 > 1800 {print $1}' | xargs -r kill -9 2>/dev/null
done ) &
( while sleep 60; do nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader; done ) >> $RUN_DIR/gpu.csv &
wait $PID; rc=$?
echo "$(date -u +%FT%TZ) train exit $rc" | tee -a $RUN_DIR/times.txt
tail -n 60 $RUN_DIR/train.log
exit $rc
