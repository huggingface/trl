"""CPU check: the W1 overrides parse into SkyRLTrainConfig, and the prompt renders as SHARED_W1 rev. 2 expects."""

import sys

import pandas as pd
from skyrl.train.config import SkyRLTrainConfig
from transformers import AutoTokenizer


for path in sys.argv[1:]:
    args = [l.strip() for l in open(path) if l.strip() and not l.strip().startswith("#")]
    args += [
        "data.train_data=['/fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/w1/train.parquet']",
        "data.val_data=[]",
    ]
    cfg = SkyRLTrainConfig.from_cli_overrides(args)
    g, a = cfg.generator, cfg.trainer.algorithm
    print(
        path.split("/")[-1],
        "chat_template_kwargs",
        dict(g.chat_template_kwargs),
        "max_gen",
        g.sampling_params.max_generate_length,
        "loss",
        a.policy_loss_type,
        a.loss_reduction,
        "kl",
        a.use_kl_loss,
        "wd",
        cfg.trainer.policy.optimizer_config.weight_decay,
        "prompt_len",
        cfg.trainer.max_prompt_length,
        "micro",
        cfg.trainer.micro_train_batch_size_per_gpu,
    )

tok = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path)
df = pd.read_parquet("/fsx/amine_dirhoussi/stable-bench-runs/skyrl/data/w1/train.parquet")
msgs = list(df["prompt"].iloc[0])
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **dict(g.chat_template_kwargs))
print(repr(text[-160:]))
assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "unexpected prompt ending"
lens = [
    len(
        tok.apply_chat_template(
            list(m), add_generation_prompt=True, tokenize=True, return_dict=False, **dict(g.chat_template_kwargs)
        )
    )
    for m in df["prompt"]
]
print(
    "prompt tokens: max",
    max(lens),
    "n>2048",
    sum(l > 2048 for l in lens),
    "n>1024",
    sum(l > 1024 for l in lens),
    "of",
    len(lens),
)
print("CONFIG CHECK OK")
