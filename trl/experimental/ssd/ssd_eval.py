# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "vllm",
#     "huggingface_hub",
#     "livecodebench @ git+https://github.com/LiveCodeBench/LiveCodeBench.git",
# ]
# ///

"""
LiveCodeBench v6 evaluation script.

Generates completions with vLLM at a configurable decoding setting and scores them with LiveCodeBench's official
``codegen_metrics`` (pass@k via sandboxed test execution). The default decoding configuration matches Table 3 of
*Embarrassingly Simple Self-Distillation Improves Code Generation* (Zhang et al., 2026), making this script suitable
for evaluating SSD-trained checkpoints alongside their base models.

Example — evaluate the base Qwen3-4B-Instruct-2507 on the v6 delta (the new problems released in v6):

```bash
python trl/experimental/ssd/ssd_eval.py \\
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \\
    --temperature 1.1 \\
    --top_k 20 \\
    --top_p 0.8 \\
    --n 1 \\
    --output_file outputs/qwen3_4b_base_lcb_v6.json
```

Evaluate an SSD-trained checkpoint with the same decoding configuration:

```bash
python trl/experimental/ssd/ssd_eval.py \\
    --model_name_or_path outputs/ssd-qwen3-4b-instruct \\
    --temperature 1.1 \\
    --top_k 20 \\
    --top_p 0.8 \\
    --n 5 \\
    --output_file outputs/qwen3_4b_ssd_lcb_v6.json
```
"""

# ruff: noqa: T201

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from huggingface_hub import hf_hub_download
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from trl import TrlParser


# LiveCodeBench stores new problems added in each release version in a separate JSONL file:
# ``test.jsonl`` = v1, ``test2.jsonl`` = v2, …, ``test6.jsonl`` = v6 (the delta added in release v6).
# The paper's "LCB v6" refers to exactly this delta.
LCB_REPO = "livecodebench/code_generation_lite"
LCB_V6_FILE = "test6.jsonl"


SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will "
    "generate a correct Python program that matches the specification and passes all tests."
)
FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
    "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
    "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)
FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code "
    "within delimiters."
)

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class SSDEvalArguments:
    model_name_or_path: str = field(metadata={"help": "Model path or Hub repo id to evaluate."})
    output_file: str = field(metadata={"help": "Path to write generations + metrics JSON."})
    temperature: float = field(default=1.1, metadata={"help": "Evaluation-time sampling temperature (T_eval)."})
    top_k: int = field(default=20, metadata={"help": "Evaluation-time top-k truncation."})
    top_p: float = field(default=0.8, metadata={"help": "Evaluation-time top-p (nucleus) truncation."})
    n: int = field(default=1, metadata={"help": "Number of samples per problem (pass@1 uses 1, pass@5 uses 5)."})
    max_tokens: int = field(default=32768, metadata={"help": "Maximum tokens to generate per completion."})
    max_model_len: int = field(default=65536, metadata={"help": "vLLM max model length."})
    gpu_memory_utilization: float = field(default=0.9, metadata={"help": "vLLM GPU memory ratio."})
    tensor_parallel_size: int = field(default=1, metadata={"help": "vLLM tensor parallel size."})
    dtype: str = field(default="bfloat16", metadata={"help": "vLLM model dtype."})
    start_date: str | None = field(
        default=None, metadata={"help": "Keep only problems with contest_date >= YYYY-MM-DD."}
    )
    end_date: str | None = field(
        default=None, metadata={"help": "Keep only problems with contest_date <= YYYY-MM-DD."}
    )
    max_problems: int | None = field(
        default=None, metadata={"help": "Evaluate at most N problems (useful for quick smoke tests)."}
    )
    difficulty: str | None = field(
        default=None, metadata={"help": "Filter to a single difficulty: 'easy', 'medium', or 'hard'."}
    )
    timeout: int = field(default=6, metadata={"help": "Per-test execution timeout in seconds."})
    num_process_evaluate: int = field(default=8, metadata={"help": "Parallel processes for sandboxed evaluation."})
    seed: int = field(default=0, metadata={"help": "vLLM sampling seed."})


def _build_prompt(tokenizer, question_content: str, starter_code: str) -> str:
    """Build the LCB prompt and apply the model's chat template."""
    body = f"### Question:\n{question_content}\n\n"
    if starter_code:
        body += f"### Format: {FORMATTING_WITH_STARTER}\n"
        body += f"```python\n{starter_code}\n```\n\n"
    else:
        body += f"### Format: {FORMATTING_WITHOUT_STARTER}\n"
        body += "```python\n# YOUR CODE HERE\n```\n\n"
    body += "### Answer: (use the provided format with backticks)\n\n"
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": body},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_code(text: str) -> str:
    """Extract the first Python code fence from a model completion."""
    match = CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: if the model returned bare code without a fence, return the whole thing.
    return text.strip()


def _load_lcb_v6_problems(args: SSDEvalArguments):
    """Load LiveCodeBench v6 problems, filter by date/difficulty/count, and return as a list."""
    jsonl_path = hf_hub_download(repo_id=LCB_REPO, filename=LCB_V6_FILE, repo_type="dataset")
    with open(jsonl_path) as f:
        problems = [CodeGenerationProblem(**json.loads(line)) for line in f]

    if args.start_date is not None:
        start = datetime.fromisoformat(args.start_date)
        problems = [p for p in problems if p.contest_date >= start]
    if args.end_date is not None:
        end = datetime.fromisoformat(args.end_date)
        problems = [p for p in problems if p.contest_date <= end]
    if args.difficulty is not None:
        problems = [p for p in problems if p.difficulty.value == args.difficulty]
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    return problems


def main():
    parser = TrlParser(SSDEvalArguments)
    (args,) = parser.parse_args_and_config()

    problems = _load_lcb_v6_problems(args)
    print(f"Evaluating {len(problems)} problems from LiveCodeBench v6")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    prompts = [_build_prompt(tokenizer, p.question_content, p.starter_code) for p in problems]

    llm = LLM(
        model=args.model_name_or_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)

    generations_list = []
    for out in outputs:
        code_list = [_extract_code(o.text) for o in out.outputs]
        generations_list.append(code_list)

    samples_list = [p.get_evaluation_sample() for p in problems]
    k_list = [1] if args.n == 1 else [1, args.n]
    metrics, results, _metadata = codegen_metrics(
        samples_list,
        generations_list,
        k_list=k_list,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )

    # Break down pass@1 by difficulty for easier comparison with the paper's tables. `codegen_metrics`
    # returns a nested `detail` dict keyed by problem index.
    per_problem_pass1 = metrics["detail"]["pass@1"]
    per_difficulty = {}
    for difficulty in ("easy", "medium", "hard"):
        idxs = [i for i, p in enumerate(problems) if p.difficulty.value == difficulty]
        if not idxs:
            continue
        per_difficulty[difficulty] = {
            "num_problems": len(idxs),
            "pass@1": sum(per_problem_pass1[i] for i in idxs) / len(idxs),
        }

    summary = {
        "model": args.model_name_or_path,
        "num_problems": len(problems),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "n": args.n,
        "metrics": {k: float(v) for k, v in metrics.items() if k != "detail"},
        "per_difficulty": per_difficulty,
    }

    detail = [
        {
            "question_id": p.question_id,
            "difficulty": p.difficulty.value,
            "code_list": code_list,
            "pass@1": per_problem_pass1[i],
        }
        for i, (p, code_list) in enumerate(zip(problems, generations_list, strict=False))
    ]

    with open(args.output_file, "w") as f:
        json.dump({"summary": summary, "detail": detail}, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {args.output_file}")


if __name__ == "__main__":
    main()
