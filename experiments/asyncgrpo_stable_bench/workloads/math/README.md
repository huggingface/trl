# W1 math workload

Shared by every library. Config values live in [configs/SHARED_W1.md](../../configs/SHARED_W1.md).

| file | what |
|---|---|
| `data/train.parquet` | DAPO-Math-17k, deduplicated on the problem text, shuffled with seed 42. Columns `problem`, `answer`. |
| `data/eval_{aime24,aime25,math500}.parquet` | `HuggingFaceH4/aime_2024`, `math-ai/aime25`, `HuggingFaceH4/MATH-500`. Columns `problem`, `answer`. |
| `reward.py` | `compute_score(completion, answer) -> 1.0 / 0.0` and the prompt template `PROMPT`. Needs only `math-verify[antlr4_13_2]`. `python reward.py` runs its self-test. |
| `prepare.py`, `prepare.sbatch` | rebuild `data/` (hopper-cpu job). |
| `eval.py`, `eval.sbatch` | evaluate HF checkpoints with vLLM and append rows to `eval.jsonl` (schema in `collect/SCHEMA.md`). |

## Using it from another library

- Prompt: one user message, `PROMPT.format(problem=row.problem)`, rendered with the model's chat template at its defaults (for Qwen3.5 that means thinking on).
- Reward: `from reward import compute_score` (put this directory on `PYTHONPATH`) and score the **raw decoded completion** (thinking included) against `answer`. It takes the content of the last `\boxed{...}` with balanced braces; an unclosed trailing `\boxed{` (truncation) scores 0. math_verify's timeouts use SIGALRM, so `compute_score` turns them off when it isn't called on the main thread.
- DAPO's original prompt wraps each problem in an "Answer: $Answer" instruction; `prepare.py` strips it (and asserts the wrapper is present on every row), so `problem` is the bare problem.

## Eval

```bash
# every checkpoint-<step> under a run dir, one per GPU, rows appended to OUT
b200-queue --gres=gpu:4 workloads/math/eval.sbatch <run_dir> <results/raw/<run_id>/eval.jsonl> [python with vllm]
# or one checkpoint
python workloads/math/eval.py --model <ckpt dir> --step 50 --out eval.jsonl
```

AIME24/25 report avg@16 and MATH-500 avg@1, temperature 1.0, top_p 1.0, max 8192 new tokens, max model length 2048 + max tokens = 10240, vLLM seed 42. `--samples-out` writes every completion with its score.
