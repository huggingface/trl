"""Base-model completion lengths on W1 train prompts, to choose max completion length and enable_thinking.

    python length_probe.py --thinking {true,false} --start 0 --count 128 --out shard.jsonl   # one GPU per shard
    python length_probe.py --aggregate shard*.jsonl                                          # stats per cap

Samples 8 completions per prompt with a 32768-token cap (T 1.0, top_p 1.0, seed 42). A completion cut at a smaller
cap L is the same sample truncated at L tokens, so one run gives the truncation rate, reward and zero-std group fraction
at every cap up to 32768.
"""

import argparse
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
CAPS = (4096, 8192, 16384, 32768)


def sample(args):
    import pandas as pd
    from reward import PROMPT, compute_score
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(os.path.join(HERE, "data", "train.parquet")).iloc[args.start : args.start + args.count]
    llm = LLM(args.model, max_model_len=2048 + max(CAPS), seed=42, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    msgs = [[{"role": "user", "content": PROMPT.format(problem=q)}] for q in df["problem"]]
    params = SamplingParams(n=8, temperature=1.0, top_p=args.top_p, top_k=args.top_k, max_tokens=max(CAPS), seed=42)
    outs = llm.chat(msgs, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": args.thinking == "true"})
    with open(args.out, "w") as f:
        for i, (out, answer) in enumerate(zip(outs, df["answer"], strict=True)):
            for c in out.outputs:
                n, stop = len(c.token_ids), c.finish_reason == "stop"
                tail = c.token_ids[-2000:]
                row = {
                    "prompt": args.start + i,
                    "len": n,
                    "stop": stop,
                    "has_think_close": "</think>" in c.text,
                    # repetition check: share of distinct 20-grams in the last 2000 tokens (low = looping)
                    "tail_distinct_20grams": len({tuple(tail[j : j + 20]) for j in range(len(tail) - 19)})
                    / max(1, len(tail) - 19),
                    "text_tail": c.text[-400:],
                }
                for cap in CAPS:
                    text = c.text if n <= cap else tok.decode(c.token_ids[:cap], skip_special_tokens=True)
                    row[f"trunc@{cap}"] = not (stop and n <= cap)
                    row[f"reward@{cap}"] = compute_score(text, answer)
                f.write(json.dumps(row) + "\n")


def aggregate(paths):
    rows = [json.loads(line) for p in paths for line in open(p)]
    lens = np.array([r["len"] for r in rows])
    looping = (
        [r["tail_distinct_20grams"] < 0.5 for r in rows if not r["stop"]] if "tail_distinct_20grams" in rows[0] else []
    )
    res = {
        "n": len(rows),
        "frac_unfinished_looping": float(np.mean(looping)) if looping else None,
        "prompts": len({r["prompt"] for r in rows}),
        "frac_with_</think>": np.mean([r["has_think_close"] for r in rows]),
    }
    for cap in CAPS:
        capped = np.minimum(lens, cap)
        groups = {}
        for r in rows:
            groups.setdefault(r["prompt"], []).append(r[f"reward@{cap}"])
        res[f"cap{cap}"] = {
            "truncated": float(np.mean([r[f"trunc@{cap}"] for r in rows])),
            "len_p50": float(np.median(capped)),
            "len_p90": float(np.percentile(capped, 90)),
            "reward_mean": float(np.mean([r[f"reward@{cap}"] for r in rows])),
            "frac_zero_std_groups": float(np.mean([len(set(g)) == 1 for g in groups.values()])),
        }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--thinking", choices=["true", "false"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--out")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--aggregate", nargs="*")
    args = p.parse_args()
    aggregate(args.aggregate) if args.aggregate else sample(args)
