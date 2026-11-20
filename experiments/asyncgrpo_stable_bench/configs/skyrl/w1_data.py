"""Convert workloads/math/data/train.parquet (problem, answer) to SkyRL's dataset format (prompt messages, env_class,
reward_spec). Same prompt as workloads/math/reward.py::PROMPT; the chat template is applied by SkyRL's generator."""

import os
import sys

import pandas as pd


BENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BENCH, "workloads", "math"))
from reward import PROMPT  # noqa: E402


OUT = "/fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/w1"
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(os.path.join(BENCH, "workloads", "math", "data", "train.parquet"))
rows = [
    {
        "data_source": "dapo_math_17k",
        "prompt": [{"role": "user", "content": PROMPT.format(problem=p)}],
        "env_class": "w1_math",
        "reward_spec": {"method": "rule", "ground_truth": a},
        "extra_info": {"index": i},
    }
    for i, (p, a) in enumerate(zip(df["problem"], df["answer"], strict=False))
]
pd.DataFrame(rows).to_parquet(os.path.join(OUT, "train.parquet"), index=False)
print(len(rows), "rows ->", OUT, rows[0]["prompt"][0]["content"][-120:])
