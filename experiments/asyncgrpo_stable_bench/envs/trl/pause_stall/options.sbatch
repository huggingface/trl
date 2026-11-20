#!/bin/bash
# "Options measured" in PAUSE_STALL.md: repro.sbatch three times on one node, 10-minute arms.
#   1. 8 API servers (w1 default): comp8 comp4 comp2 comp1 comp0  -> logs/<jobid>-api8
#   2. 16 API servers: comp8                                     -> logs/<jobid>-api16
#   3. 4 API servers: comp8                                      -> logs/<jobid>-api4
#   b200-queue --nodes=1 --gres=gpu:8 --exclusive -t 150 envs/trl/pause_stall/options.sbatch
#SBATCH -J pause-options
#SBATCH --nodes=1
#SBATCH -t 02:30:00
#SBATCH -o /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/pause_stall/logs/%x-%j.out
set -uo pipefail
D=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/envs/trl/pause_stall
export MEASURE_S=600 FILL_S=120
run() {  # <api servers> <arms>
    API_SERVERS=$1 ARMS=$2 OUT_SUFFIX=-api$1 bash $D/repro.sbatch
    # the previous server must release its GPUs (and port) before the next one starts
    for _ in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
    sleep 10
}
run 8 "comp8 comp4 comp2 comp1 comp0"
run 16 "comp8"
run 4 "comp8"
