"""Survey of TermGrade (ai-and/termgrade-environments + trajectories metadata): what the tasks need and how hard they are.

python survey.py [--src /fsx/amine_dirhoussi/stable-bench-src/termgrade]   (hopper-cpu job, writes survey.json next to it)
"""

import argparse
import collections
import os
import re
import tomllib

import pandas as pd


p = argparse.ArgumentParser()
p.add_argument("--src", default="/fsx/amine_dirhoussi/stable-bench-src/termgrade")
args = p.parse_args()
env_root = os.path.join(args.src, "environments")
tasks = pd.read_json(os.path.join(env_root, "tasks.jsonl"), lines=True)
labels = pd.read_json(os.path.join(env_root, "labels.jsonl"), lines=True)
print("tasks.jsonl", tasks.shape, list(tasks.columns))
print(tasks.head(3).to_string())
print("labels.jsonl", labels.shape, list(labels.columns))
print(labels.head(6).to_string())
print("split counts", tasks["split"].value_counts().to_dict())
print("oracle_verified", tasks["oracle_verified"].value_counts().to_dict())

dirs = sorted(os.listdir(os.path.join(env_root, "environments")))
print("task dirs", len(dirs))
toml_keys, dockerfiles, net = collections.Counter(), collections.Counter(), collections.Counter()
setup_feats = collections.Counter()
cats, diffs = collections.Counter(), collections.Counter()
instr_chars, setup_chars = [], []
for d in dirs:
    t = os.path.join(env_root, "environments", d)
    cfg = tomllib.load(open(os.path.join(t, "task.toml"), "rb"))
    for sec, v in cfg.items():
        for k in v:
            toml_keys[f"{sec}.{k}"] += 1
    net[cfg.get("environment", {}).get("network_mode")] += 1
    cats[cfg.get("metadata", {}).get("category")] += 1
    diffs[cfg.get("metadata", {}).get("difficulty")] += 1
    dockerfiles[open(os.path.join(t, "environment", "Dockerfile")).read()] += 1
    s = open(os.path.join(t, "environment", "setup.sh")).read()
    setup_chars.append(len(s))
    instr_chars.append(len(open(os.path.join(t, "instruction.md")).read()))
    for feat, rx in {
        "pip": r"\bpip3? install",
        "apt": r"apt(-get)? install",
        "curl/wget": r"\b(curl|wget)\b",
        "git clone": r"git clone",
        "useradd": r"useradd|adduser",
        "systemctl/service": r"systemctl|service ",
        "writes /tests": r"/tests",
        "writes /app": r"/app",
        "chmod/chown": r"chown|chmod",
    }.items():
        if re.search(rx, s):
            setup_feats[feat] += 1
print("task.toml keys", dict(toml_keys))
print("network_mode", dict(net))
print("categories", cats.most_common())
print("difficulty", dict(diffs))
print("distinct Dockerfiles", len(dockerfiles), [c for c in dockerfiles.values()])
print("setup.sh features", dict(setup_feats))
s = pd.Series(setup_chars)
i = pd.Series(instr_chars)
print(
    "setup.sh chars p50/p90/max",
    s.median(),
    s.quantile(0.9),
    s.max(),
    "| instruction chars p50/p90/max",
    i.median(),
    i.quantile(0.9),
    i.max(),
)

traj = os.path.join(args.src, "trajectories")
trials = pd.read_parquet(os.path.join(traj, "data", "trials.parquet"))
print("trials", trials.shape, "models", trials["model"].value_counts().to_dict())
print(
    trials.iloc[0][
        [
            "task_timeout_s",
            "reaper_cap_s",
            "turn_latencies_ms",
            "phase_s",
            "status",
            "reason",
            "cap_s",
            "serving_stack",
            "served_as",
            "model_id",
        ]
    ].to_dict()
)
c = trials[trials["counts_toward_k"]]
g = c.groupby("model").agg(
    pass1=("reward", "mean"),
    n=("reward", "size"),
    dur_p50=("duration_s", "median"),
    dur_p90=("duration_s", lambda x: x.quantile(0.9)),
    eps_p50=("n_episodes", "median"),
    eps_p90=("n_episodes", lambda x: x.quantile(0.9)),
    prompt_tok_p50=("prompt_tokens", "median"),
    out_tok_p50=("output_tokens", "median"),
    out_tok_p90=("output_tokens", lambda x: x.quantile(0.9)),
    summ=("summarizations", "mean"),
)
print(g.to_string())
ph = c["phase_s"].dropna()
if len(ph):
    keys = collections.Counter(k for d in ph for k in (d or {}))
    print("phase keys", keys)
    for k in keys:
        v = pd.Series([d.get(k) for d in ph if d and d.get(k) is not None], dtype=float)
        print(f"  phase {k}: p50 {v.median():.1f} p90 {v.quantile(0.9):.1f}")
print("task_timeout_s values", c["task_timeout_s"].astype(str).value_counts().head(5).to_dict())

# per task pass rate for the qwen (Qwen3.6-27B) and gemma (31B, no thinking) rows, the two closest to our models
lab = labels.pivot_table(index="task", columns="model", values="pass_rate")
print("labels models", list(lab.columns))
for m in lab.columns:
    v = lab[m].dropna()
    print(
        f"  {m}: n {len(v)} mean {v.mean():.3f} frac0 {(v == 0).mean():.3f} frac1 {(v == 1).mean():.3f} frac in [0.2,0.8] {((v >= 0.2) & (v <= 0.8)).mean():.3f}"
    )
summary = tasks.merge(lab.reset_index(), on="task", how="left")
summary.to_csv(os.path.join(args.src, "task_summary.csv"), index=False)
print("wrote task_summary.csv", summary.shape)
