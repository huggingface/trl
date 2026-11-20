"""Rendered-prompt check for W1 rev. 2: chat template with the config's apply_chat_template_kwargs, as verl's agent loop does."""

import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer


cfg = OmegaConf.load(
    "/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/configs/verl/w1-S.yaml"
)
kw = OmegaConf.to_container(cfg.data.apply_chat_template_kwargs)
tok = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path)
df = pd.read_parquet(cfg.data.train_files)
text = tok.apply_chat_template(list(df["prompt"].iloc[0]), add_generation_prompt=True, tokenize=False, **kw)
print("kwargs", kw)
print("tail", repr(text[-80:]))
print("ends with required suffix:", text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
lens = [
    len(
        tok(
            tok.apply_chat_template(list(p), add_generation_prompt=True, tokenize=False, **kw),
            add_special_tokens=False,
        )["input_ids"]
    )
    for p in df["prompt"]
]
s = pd.Series(lens)
print(f"prompt tokens: max {s.max()}, >1024: {(s > 1024).sum()}, >2048: {(s > 2048).sum()} of {len(s)}")
