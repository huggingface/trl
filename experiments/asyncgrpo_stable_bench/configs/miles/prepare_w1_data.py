"""workloads/math/data/train.parquet (problem, answer) -> the miles input format: a `messages` column with the single
user message of SHARED_W1 (reward.PROMPT) and a `label` column. miles applies the model's chat template to `messages`
(--apply-chat-template, no template kwargs = the template defaults). Row order (the seed-42 shuffle) is kept.

    python prepare_w1_data.py <out.parquet>
"""

import os
import sys

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
MATH = os.path.join(HERE, "..", "..", "workloads", "math")
sys.path.insert(0, MATH)
from reward import PROMPT  # noqa: E402


df = pd.read_parquet(os.path.join(MATH, "data", "train.parquet"))
out = pd.DataFrame(
    {
        "messages": [[{"role": "user", "content": PROMPT.format(problem=p)}] for p in df["problem"]],
        "label": df["answer"],
    }
)
out.to_parquet(sys.argv[1], index=False)
print(len(out), "rows ->", sys.argv[1])
print(out["messages"].iloc[0][0]["content"][-200:], "| label:", out["label"].iloc[0])
