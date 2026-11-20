"""Sample completions with vLLM (top-20 processed logprobs per token) and save ids + logprobs. Usage: gen.py <sampler ckpt> <out.pt>"""

import os
import sys

import pandas as pd
import torch
from vllm import LLM, SamplingParams


BENCH = "/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench"
sys.path.insert(0, os.path.join(BENCH, "workloads", "math"))
from reward import PROMPT  # noqa: E402


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(os.path.join(BENCH, "workloads", "math", "data", "train.parquet")).sample(48, random_state=0)
    llm = LLM(
        model=ckpt,
        tokenizer="Qwen/Qwen3.5-2B",
        dtype="bfloat16",
        max_model_len=6144,
        gpu_memory_utilization=0.6,
        logprobs_mode="processed_logprobs",
        max_logprobs=20,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(problem=p)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in df["problem"]
    ]
    outs = llm.generate(
        prompts, SamplingParams(temperature=1.0, top_p=1.0, top_k=0, max_tokens=4096, logprobs=20, seed=0)
    )
    rows = []
    for o in outs:
        c = o.outputs[0]
        rows.append(
            {
                "prompt_ids": list(o.prompt_token_ids),
                "ids": list(c.token_ids),
                "top": [
                    sorted(((t, lp.logprob) for t, lp in d.items() if lp.rank <= 20), key=lambda x: -x[1])[:20]
                    for d in c.logprobs
                ],
                "sampled_lp": [d[t].logprob for t, d in zip(c.token_ids, c.logprobs, strict=False)],
            }
        )
    torch.save(rows, out)
    print("saved", len(rows), "completions,", sum(len(r["ids"]) for r in rows), "tokens")


if __name__ == "__main__":
    main()
