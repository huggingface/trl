"""Evaluate one HF checkpoint on AIME24 / AIME25 (avg@16) and MATH-500 (avg@1) with vLLM; append a row to eval.jsonl.

    python eval.py --model <hf checkpoint dir> --step 50 --out results/raw/<run_id>/eval.jsonl

Sampling matches W1 training (SHARED_W1 rev. 2): temperature 1.0, top_p 1.0, max 16384 new tokens, enable_thinking=False.
"""

import argparse
import glob
import json
import os
import time

import pandas as pd
from reward import PROMPT, compute_score


HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {"aime24": 16, "aime25": 16, "math500": 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF checkpoint dir (or hub id)")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--out", required=True, help="eval.jsonl to append to")
    p.add_argument(
        "--time", type=float, default=None, help="wall clock when the checkpoint was saved (default: mtime)"
    )
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--enable-thinking", choices=["true", "false"], default="false", help="passed to the chat template")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-out", default=None, help="optional jsonl with every completion and its score")
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    t_saved = args.time
    if t_saved is None:
        weights = glob.glob(os.path.join(args.model, "*.safetensors")) or glob.glob(
            os.path.join(args.model, "config.json")
        )
        t_saved = max(os.path.getmtime(f) for f in weights) if weights else None

    llm = LLM(
        args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.tp,
        max_model_len=2048 + args.max_tokens,  # max prompt 2048 (SHARED_W1 rev. 1)
        seed=args.seed,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    row = {"step": args.step, "time": t_saved}
    lengths, records = [], []
    for name, n in SETS.items():
        df = pd.read_parquet(os.path.join(HERE, "data", f"eval_{name}.parquet"))
        messages = [[{"role": "user", "content": PROMPT.format(problem=q)}] for q in df["problem"]]
        params = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)
        t0 = time.time()
        kwargs = {} if args.enable_thinking is None else {"enable_thinking": args.enable_thinking == "true"}
        outs = llm.chat(messages, params, use_tqdm=False, chat_template_kwargs=kwargs)
        scores = []
        for (problem, answer), out in zip(df[["problem", "answer"]].itertuples(index=False), outs, strict=True):
            for c in out.outputs:
                s = compute_score(c.text, answer)
                scores.append(s)
                lengths.append(len(c.token_ids))
                if args.samples_out:
                    records.append(
                        {
                            "set": name,
                            "problem": problem,
                            "answer": answer,
                            "text": c.text,
                            "score": s,
                            "len": len(c.token_ids),
                            "finish_reason": c.finish_reason,
                        }
                    )
        row[f"{name}_avg{n}"] = sum(scores) / len(scores)
        print(f"{name}: avg@{n} = {row[f'{name}_avg{n}']:.4f} over {len(scores)} samples in {time.time() - t0:.0f}s")
    row["mean_len"] = sum(lengths) / len(lengths)
    row["model"] = args.model
    row["enable_thinking"] = args.enable_thinking
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    if args.samples_out:
        with open(args.samples_out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
