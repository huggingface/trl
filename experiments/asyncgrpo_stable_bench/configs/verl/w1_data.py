"""Convert workloads/math/data/{train,eval_math500}.parquet (problem, answer) into verl's RL parquet format.

verl renders `prompt` (a chat message list) with the model's chat template; the user message is the shared W1 prompt.
Usage: python configs/verl/w1_data.py <out_dir>
"""

import os
import sys

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "workloads", "math", "data")
sys.path.insert(0, os.path.join(HERE, "..", "..", "workloads", "math"))
from reward import PROMPT  # noqa: E402


def convert(name, out):
    df = pd.read_parquet(os.path.join(SRC, f"{name}.parquet"))
    rows = [
        {
            "data_source": "w1_math",
            "prompt": [{"role": "user", "content": PROMPT.format(problem=p)}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": a},
            "extra_info": {"index": i},
        }
        for i, (p, a) in enumerate(zip(df["problem"], df["answer"], strict=False))
    ]
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"{name}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    convert("train", os.path.join(out_dir, "train.parquet"))
    convert("eval_math500", os.path.join(out_dir, "val.parquet"))
