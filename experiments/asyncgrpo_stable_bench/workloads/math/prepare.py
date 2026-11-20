"""Build the W1 math data: DAPO-Math-17k (train) and AIME24 / AIME25 / MATH-500 (eval), as parquet with `problem`, `answer`."""

import os
import re

import datasets


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# DAPO-Math-17k wraps every problem in a fixed instruction asking for an "Answer:" line. We keep only the problem text,
# since the shared prompt asks for \boxed{} instead.
DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def dapo_problem(prompt):
    text = prompt[0]["content"]
    assert text.startswith(DAPO_PREFIX) and text.endswith(DAPO_SUFFIX), text[:300]
    return text[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)].strip()


def main():
    os.makedirs(OUT, exist_ok=True)

    ds = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    print("DAPO raw:", ds)
    df = ds.to_pandas()
    df = df.assign(
        problem=df["prompt"].map(dapo_problem), answer=df["reward_model"].map(lambda r: str(r["ground_truth"]).strip())
    )[["problem", "answer"]]
    n_raw = len(df)
    df = df.drop_duplicates(subset="problem", keep="first")
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"DAPO: {n_raw} rows -> {len(df)} unique problems")
    df.to_parquet(os.path.join(OUT, "train.parquet"), index=False)

    evals = {
        "aime24": ("HuggingFaceH4/aime_2024", None, "train"),
        "aime25": ("math-ai/aime25", None, "test"),
        "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    }
    for name, (repo, cfg, split) in evals.items():
        ds = datasets.load_dataset(repo, cfg, split=split)
        print(name, ds)
        df = ds.to_pandas()
        df = df.assign(answer=df["answer"].map(lambda a: re.sub(r"^0+(?=\d)", "", str(a).strip())))[
            ["problem", "answer"]
        ]
        df.to_parquet(os.path.join(OUT, f"eval_{name}.parquet"), index=False)
        print(f"{name}: {len(df)} rows, first answer {df['answer'].iloc[0]!r}")


if __name__ == "__main__":
    main()
