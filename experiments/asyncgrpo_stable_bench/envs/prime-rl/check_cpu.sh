#!/bin/bash
# CPU-side checks for prime-rl (hopper-cpu job): model downloads, config dry-runs, Qwen3.5 renderer parity, W1 taskset.
#SBATCH --job-name=prl-check
#SBATCH --partition=hopper-cpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/logs/%x-%j.out
set -uo pipefail
SRC=/fsx/amine_dirhoussi/stable-bench-src/prime-rl
export BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
source /fsx/amine_dirhoussi/envs/bench-prime-rl/bin/activate
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub
export PYTHONPATH=$BENCH/configs/prime-rl
DRY=/fsx/amine_dirhoussi/stable-bench-runs/prime-rl/dryrun
cd "$SRC"

if [ "${DOWNLOAD:-1}" = 1 ]; then
    for m in Qwen/Qwen3.5-2B PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT Qwen/Qwen3-1.7B; do
        hf download "$m" --quiet > /dev/null && echo "downloaded $m"
    done
    python - <<'EOF'
from datasets import load_dataset
for name, kw in [("PrimeIntellect/Reverse-Text-RL", {"split": "train"}),
                 ("mikasenghaas/Sanity-Test-R1D-1.5B", {"name": "default", "split": "train"})]:
    ds = load_dataset(name, **kw); print(name, len(ds), ds.column_names)
EOF
    cat "$(hf download Qwen/Qwen3.5-2B generation_config.json --quiet)"
fi

for cfg in ${CFGS:-examples/basic/reverse-text/rl.toml "$BENCH/configs/prime-rl/recipe-hendrycks-sanity.toml" "$BENCH/configs/prime-rl/w1-S.toml"}; do
    name=$(basename "$(dirname "$cfg")")-$(basename "$cfg" .toml)
    echo "=== dry-run $cfg"
    rl @ "$cfg" --output-dir "$DRY" --run.name "$name" --clean --no-monitors.prime --dry-run 2>&1 | tail -n 15
    python - "$DRY/$name" <<'EOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]) / "configs/latest/resolved"
for f in sorted(d.glob("*.json")):
    c = json.loads(f.read_text())
    mon = c.get("monitors", {})
    print(f.name, "monitors:", {k: (v is not None) for k, v in mon.items()} if isinstance(mon, dict) else mon)
EOF
done

python - <<'EOF'
# Qwen3.5-2B: prime-rl's renderer vs the checkpoint's own chat template at defaults, on the W1 prompt.
import os, sys
from transformers import AutoTokenizer
from renderers import create_renderer
from renderers.configs import Qwen35RendererConfig
sys.path.insert(0, os.path.join(os.environ["BENCH"], "workloads", "math"))
from reward import PROMPT
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
msgs = [{"role": "user", "content": PROMPT.format(problem="What is $1+1$?")}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ref = tok(text, add_special_tokens=False)["input_ids"]
r = create_renderer(tok, Qwen35RendererConfig())
ids = list(r.render_ids(msgs, add_generation_prompt=True))
print("renderer:", type(r).__name__, "token-for-token match with apply_chat_template:", ids == ref, len(ids), len(ref))
print("template:", repr(text))
if ids != ref:
    print("renderer:", repr(tok.decode(ids)))
EOF

if [ -f "$BENCH/workloads/math/data/train.parquet" ]; then
python - <<'EOF'
import asyncio, os
import pandas as pd
from transformers import AutoTokenizer
import bench_w1_math as m
from reward import PROMPT
ts = m.W1MathTaskset(m.W1MathConfig())
tasks = ts.load()
print("W1 tasks:", len(tasks), "first prompt:", repr(tasks[0].data.prompt[:120]), "answer:", tasks[0].data.answer)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
df = pd.read_parquet(m.W1MathConfig().data_path)
texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=p)}], add_generation_prompt=True, tokenize=False) for p in df["problem"]]
s = pd.Series([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])
print("prompt tokens:", s.describe().to_dict(), "over 1024:", int((s > 1024).sum()))
EOF
fi
