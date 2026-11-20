"""Token length of every W1 prompt (train + eval) under a model's chat template at its defaults.

python prompt_lengths.py [model]   # default Qwen/Qwen3.5-2B
"""

import os
import sys

import numpy as np
import pandas as pd
from reward import PROMPT
from transformers import AutoTokenizer


HERE = os.path.dirname(os.path.abspath(__file__))
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B"
tok = AutoTokenizer.from_pretrained(model)
for name in ["train", "eval_aime24", "eval_aime25", "eval_math500"]:
    df = pd.read_parquet(os.path.join(HERE, "data", f"{name}.parquet"))
    msgs = [[{"role": "user", "content": PROMPT.format(problem=p)}] for p in df["problem"]]
    n = np.array(
        [len(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True, return_dict=False)) for m in msgs]
    )
    print(
        f"{name}: n={len(n)} max={n.max()} p50={np.median(n):.0f} p99={np.percentile(n, 99):.0f} "
        f">1024: {(n > 1024).sum()} >2048: {(n > 2048).sum()}"
    )
