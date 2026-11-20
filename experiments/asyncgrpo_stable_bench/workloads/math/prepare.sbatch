#!/bin/bash
#SBATCH -J math-prep
#SBATCH -p hopper-cpu
#SBATCH --qos=low
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/trl/prep/%x-%j.out
# Any python with `datasets` and `math-verify` works; PY defaults to the TRL bench venv.
set -euo pipefail
PY=${PY:-/fsx/amine_dirhoussi/envs/bench-trl/bin/python}
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_DATASETS_CACHE=/fsx/amine_dirhoussi/stable-bench-hub/datasets
cd /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/math
$PY reward.py
$PY prepare.py
$PY - <<'PY'
import pandas as pd
from reward import compute_score
for f in ["train", "eval_aime24", "eval_aime25", "eval_math500"]:
    df = pd.read_parquet(f"data/{f}.parquet")
    print(f, len(df), list(df.columns), "| answer self-score:", sum(compute_score(f"\\boxed{{{a}}}", a) for a in df["answer"][:500]), "/", min(500, len(df)))
print(pd.read_parquet("data/train.parquet").head(3).to_string()[:1500])
PY
