#!/bin/bash
# Export every DCP checkpoint of the given prime-rl runs to complete HF bf16 dirs (<run>/hf/step_N, SHARED_W1 rev. 6):
#   1. envs/prime-rl/convert_dcp_to_bf16.py: bench copy of prime-rl's converter with the tied lm_head fix (NOTES.md #19)
#   2. envs/prime-rl/complete_hf.py: copy any tensor the export lacks (MTP, vision) from the base checkpoint, check config
#   3. envs/prime-rl/vllm_load_check.py on the last step: vLLM loads it as Qwen3_5ForConditionalGeneration and answers
# Usage: b200-queue --gres=gpu:1 [-w <node already in use>] launch/prime-rl/export.sbatch <run_id> [<run_id> ...]
#SBATCH --job-name=prl-export
#SBATCH --partition=hopper-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out

set -uo pipefail
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
RUNS=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl
source /fsx/amine_dirhoussi/envs/bench-prime-rl/bin/activate
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONPATH=$BENCH/configs/prime-rl
PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))
cd /fsx/amine_dirhoussi/stable-bench-src/prime-rl
rc=0
for run in "$@"; do
    last=
    for step_dir in "$RUNS/$run"/checkpoints/step_*; do
        [ -d "$step_dir/trainer" ] || continue
        step=${step_dir##*_}
        out=$RUNS/$run/hf/step_$step
        if [ ! -f "$out/config.json" ]; then
            if torchrun --nproc-per-node 1 --master-port $((PORT + 7)) "$BENCH/envs/prime-rl/convert_dcp_to_bf16.py" "$step_dir" "$out"; then
                echo "exported $run step $step -> $out"
            else
                echo "export of $run step $step failed"; rc=1; continue
            fi
        fi
        python "$BENCH/envs/prime-rl/complete_hf.py" "$out" || rc=1
        last=$out
    done
    [ -n "$last" ] && { python "$BENCH/envs/prime-rl/vllm_load_check.py" "$last" 2>&1 | grep -E "vLLM architecture|greedy|boxed|Error|error" || rc=1; }
done
exit $rc
