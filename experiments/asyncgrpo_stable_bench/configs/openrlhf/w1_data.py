"""Write the W1 train set in the shape OpenRLHF's PromptDataset reads: `prompt` (the shared user message already rendered
with the model's chat template) and `label` (the gold answer, passed to the reward as `labels`).

OpenRLHF's `--data.apply_chat_template` calls `apply_chat_template(chat, tokenize=False, add_generation_prompt=True)`
with no way to pass template kwargs such as `enable_thinking`. So we render here with the same call (the model's
AutoProcessor, which is what OpenRLHF uses for a VLM checkpoint) plus any kwargs, and the launch runs without
`--data.apply_chat_template`: the rendered string goes to vLLM as is.

    python configs/openrlhf/w1_data.py <model dir> <out.parquet> [--enable-thinking true|false]
"""

import argparse
import os
import sys

import pandas as pd
from transformers import AutoProcessor


BENCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BENCH, "workloads", "math"))
from reward import PROMPT  # noqa: E402


p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("out")
p.add_argument("--enable-thinking", choices=["true", "false"], default=None, help="default: the template's default")
args = p.parse_args()

kwargs = {} if args.enable_thinking is None else {"enable_thinking": args.enable_thinking == "true"}
processor = AutoProcessor.from_pretrained(args.model)
df = pd.read_parquet(os.path.join(BENCH, "workloads", "math", "data", "train.parquet"))
prompts = [
    processor.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(problem=q)}], tokenize=False, add_generation_prompt=True, **kwargs
    )
    for q in df["problem"]
]
n_tok = [len(processor.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in prompts]
pd.DataFrame({"prompt": prompts, "label": df["answer"]}).to_parquet(args.out)
print(
    f"{len(df)} rows -> {args.out}  template kwargs {kwargs}  prompt tokens max {max(n_tok)}, >2048: {sum(n > 2048 for n in n_tok)}"
)
print("example:", repr(prompts[0][-200:]))
