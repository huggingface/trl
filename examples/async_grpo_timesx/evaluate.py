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
# ]
# ///

"""Score a TimesX checkpoint's point forecasts on the held-out `test` split.

AsyncGRPOTrainer has no `eval_dataset` hook (see `data.py`'s docstring), so this is a standalone script instead of
periodic in-loop validation: run it after training, pointed at a saved checkpoint (or the base model, for a
before/after comparison). It loads the model into an offline vLLM engine, generates against every held-out window,
and reports the same metrics `async_grpo_timesx.py` trains on -- MASE and format validity -- plus per-question detail.

Example:

```bash
python examples/async_grpo_timesx/evaluate.py --model_name_or_path async_grpo_timesx/checkpoint-591
```
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import DEFAULT_DOMAIN, DEFAULT_REVISION, load_timesx_split, render_prompt
from vllm import LLM, SamplingParams

from trl import TrlParser


def parse_forecast(text: str, expected_length: int) -> list[float] | None:
    """Parse the completion's last non-empty line as `expected_length` space/comma-separated numbers.

    Ported verbatim from `async_grpo_timesx.py` so this script stays a self-contained evaluator, not a training
    script's dependency.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [part for part in re.split(r"[,\s\[\]]+", lines[-1]) if part]
    if len(parts) != expected_length:
        return None
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    return values if all(math.isfinite(value) for value in values) else None


def mase(forecast: list[float], future_values: list[float], past_values: list[float]) -> float:
    """Mean Absolute Scaled Error, scaled by the history's own lag-1 naive error."""
    naive_scale = sum(abs(b - a) for a, b in zip(past_values, past_values[1:], strict=False)) / (len(past_values) - 1)
    mean_absolute_error = sum(abs(f - y) for f, y in zip(forecast, future_values, strict=True)) / len(future_values)
    return mean_absolute_error / max(naive_scale, 1e-8)


@dataclass
class TimesXEvalArguments:
    model_name_or_path: str = field(metadata={"help": "Model path or Hub repo id to evaluate."})
    domain: str = field(
        default=DEFAULT_DOMAIN, metadata={"help": "TimesX domain, e.g. 'CommodityPrice/EnergyAndFuels'."}
    )
    revision: str = field(default=DEFAULT_REVISION, metadata={"help": "Revision of the kashif/timesx dataset."})
    max_questions: int | None = field(
        default=None, metadata={"help": "Evaluate at most N held-out windows (useful for quick smoke tests)."}
    )
    n: int = field(default=1, metadata={"help": "Forecasts sampled per window; MASE is averaged over all of them."})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})
    max_tokens: int = field(default=512, metadata={"help": "Maximum tokens to generate per completion."})
    max_model_len: int = field(default=8192, metadata={"help": "vLLM max model length."})
    gpu_memory_utilization: float = field(default=0.9, metadata={"help": "vLLM GPU memory ratio."})
    tensor_parallel_size: int = field(default=1, metadata={"help": "vLLM tensor parallel size."})
    dtype: str = field(default="bfloat16", metadata={"help": "vLLM model dtype."})
    seed: int = field(default=0, metadata={"help": "vLLM sampling seed."})
    output_file: str | None = field(
        default=None, metadata={"help": "Optional path to write per-question detail as JSON."}
    )


def main() -> None:
    parser = TrlParser(TimesXEvalArguments)
    (args,) = parser.parse_args_and_config()

    test_examples = load_timesx_split(args.domain, revision=args.revision).test
    if args.max_questions is not None:
        test_examples = test_examples[: args.max_questions]
    print(f"Evaluating {len(test_examples)} held-out windows from {args.domain!r}", flush=True)

    messages = [[{"role": "user", "content": render_prompt(example)}] for example in test_examples]

    llm = LLM(
        model=args.model_name_or_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed
    )
    outputs = llm.chat(messages, sampling_params, chat_template_kwargs={"enable_thinking": False})

    per_question = []
    for example, output in zip(test_examples, outputs, strict=True):
        mases = [
            mase(forecast, list(example.future_values), list(example.past_values))
            for completion in output.outputs
            if (forecast := parse_forecast(completion.text, len(example.future_values))) is not None
        ]
        per_question.append(
            {
                "variable": example.variable,
                "idx": example.idx,
                "valid": len(mases),
                "n": len(output.outputs),
                "mase": sum(mases) / len(mases) if mases else None,
            }
        )

    valid_format_rate = sum(q["valid"] for q in per_question) / sum(q["n"] for q in per_question)
    scored = [q["mase"] for q in per_question if q["mase"] is not None]
    summary = {
        "model": args.model_name_or_path,
        "domain": args.domain,
        "num_questions": len(test_examples),
        "valid_format_rate": valid_format_rate,
        "mean_mase": sum(scored) / len(scored) if scored else None,
        "num_scored": len(scored),
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump({"summary": summary, "detail": per_question}, f, indent=2)
        print(f"\nFull results written to {args.output_file}")


if __name__ == "__main__":
    main()
