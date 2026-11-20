#!/bin/bash
# CPU job: (1) overlay with math-verify for the shared reward (the miles image does not ship it), installed with uv
# into a --target dir that launch scripts append to PYTHONPATH; --no-deps so nothing in the image is shadowed
# (sympy comes from the image). (2) W1 train data in miles' input format.
set -euo pipefail
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
OV=/fsx/amine_dirhoussi/envs/bench-miles
rm -rf $OV
uv pip install --target $OV --no-deps "math-verify==0.8.0" "latex2sympy2_extended==1.10.2" "antlr4-python3-runtime==4.13.2"
ls $OV
mkdir -p /fsx/amine_dirhoussi/stable-bench-runs/miles/data
uv run --no-project --with pandas --with pyarrow --with "math-verify[antlr4_13_2]==0.8.0" \
    python $B/configs/miles/prepare_w1_data.py /fsx/amine_dirhoussi/stable-bench-runs/miles/data/w1_train.parquet
uv run --no-project --with "math-verify[antlr4_13_2]==0.8.0" python $B/workloads/math/reward.py
