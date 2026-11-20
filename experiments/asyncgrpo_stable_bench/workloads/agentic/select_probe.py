"""Pick the base-pass-rate probe candidates from TermGrade (writes probe_tasks.txt next to this file).

Clean split only (`unseen`: the 802 tasks ai& never trained on), setup.sh without apt / curl / wget / systemctl (those
are the slow or fragile sandbox starts, since setup.sh runs at sandbox start here), then stratified on the measured
terminus-2 pass rate of the strongest open models in the same size class: 64 tasks Qwen3.6-27B always solves and
gemma-4-31B (no thinking) solves >= 75%, and 64 tasks Qwen3.6-27B solves 50-88% (128 in all). A 9B without thinking is expected to be
well below both, so the pool leans easy.
"""

import os
import re

import pandas as pd


SRC = "/fsx/amine_dirhoussi/stable-bench-src/termgrade"
here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SRC, "task_summary.csv"))
slow = re.compile(r"apt(-get)? (update|install)|\b(curl|wget)\b|systemctl|service ")
df["setup_slow"] = [
    bool(slow.search(open(os.path.join(SRC, "environments", "environments", t, "environment", "setup.sh")).read()))
    for t in df["task"]
]
pool = df[(df["split"] == "unseen") & ~df["setup_slow"]]
print("unseen", (df["split"] == "unseen").sum(), "unseen + fast setup", len(pool))
easy = pool[(pool["qwen"] == 1.0) & (pool["gemma"] >= 0.75)]
medium = pool[(pool["qwen"] >= 0.5) & (pool["qwen"] < 1.0)]
print("easy pool", len(easy), "medium pool", len(medium))
pick = pd.concat([easy.sample(64, random_state=0), medium.sample(64, random_state=0)])
with open(os.path.join(here, "probe_tasks.txt"), "w") as f:
    f.writelines(t + "\n" for t in pick["task"])
print(pick[["task", "qwen", "gemma", "gemma_think"]].describe().to_string())
pool[["task"]].to_csv(os.path.join(here, "fast_pool.txt"), index=False, header=False)
print("fast pool written:", len(pool))
