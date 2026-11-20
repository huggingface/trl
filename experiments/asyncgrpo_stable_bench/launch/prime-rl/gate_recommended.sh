#!/bin/bash
# hopper-cpu gate for the W1-S recommended arm: starts after the shared run's queue job ends (dependency set at submit),
# then waits until TRL's trl-w1 job is RUNNING (coordinator priority call), then queues the arm through b200-queue-inner.
set -uo pipefail
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
until squeue -u "$USER" -h -n trl-w1 -t R | grep -q .; do sleep 120; done
echo "trl-w1 running at $(date -u); queueing prime-rl W1-S recommended"
cd "$B" && exec /fsx/amine_dirhoussi/bin/b200-queue-inner --nodes=1 --gres=gpu:8 --exclusive "$B/launch/prime-rl/w1.sbatch" S recommended
