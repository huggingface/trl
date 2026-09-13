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
#     "trackio",
# ]
# ///

"""AsyncGRPO training of a numeric time-series forecaster on TimesX (https://huggingface.co/papers/2607.06973).

TimesX is an eval benchmark, not a training set. The upstream repo doesn't have the paper's original 2018-2022
training split, so the train/test split here is our own leakage-avoiding stand-in and isn't comparable to the
paper's published numbers (see `data.py`). This is a demo of the recipe, not a benchmark run.

Each prompt is one TimesX rolling window: 96 historical points of a real variable (a commodity price by default)
plus its textual context (background, calendar effects, related indicators, news events). The model forecasts the
next 12 points without seeing the answer or the naive baseline it's scored against.

Reward is `1 / (1 + MASE)`. A score of 0.5 means "as good as predicting tomorrow = today", higher is better. A
malformed completion (not exactly 12 numbers) gets 0.0 from both reward functions below.

Ports thinking-machines-lab/tinker-cookbook's `recipes/forecasting` (Prophet Arena, binary markets, Brier reward) to
a numeric multi-step target instead.

Data comes from `datasets.load_dataset("kashif/timesx")`, a flattened mirror of
https://github.com/haoxin1998/TimesX-project (see that dataset's card for licensing). Only `train` is used here;
`data.load_timesx_split(...).test` is there if you want to score a checkpoint afterwards.

CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-4B \
    --max-model-len 8192 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=0 accelerate launch examples/async_grpo_timesx/async_grpo_timesx.py
"""

from __future__ import annotations

import math
import re

from data import DEFAULT_DOMAIN, load_timesx_split, render_prompt
from datasets import Dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


def parse_forecast(text: str, expected_length: int) -> list[float] | None:
    """Parse the completion's last non-empty line as `expected_length` space/comma-separated numbers.

    A 0.6B model asked for a "JSON list" reliably ignores the brackets and just writes the numbers out, so this
    parses what the model actually produces instead of what was asked for -- same reasoning as
    `examples/async_grpo_prophet_arena`'s `parse_probability`.
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


def format_reward(completions: list[list[dict[str, str]]], future_values: list[list[float]], **kwargs) -> list[float]:
    """1.0 if the completion parses to exactly as many numbers as there are future timestamps, else 0.0."""
    return [
        1.0 if parse_forecast(completion[0]["content"], len(targets)) is not None else 0.0
        for completion, targets in zip(completions, future_values, strict=True)
    ]


def mase_reward(
    completions: list[list[dict[str, str]]],
    future_values: list[list[float]],
    past_values: list[list[float]],
    **kwargs,
) -> list[float]:
    """`1 / (1 + MASE)`, or 0.0 for a malformed completion."""
    rewards = []
    for completion, targets, history in zip(completions, future_values, past_values, strict=True):
        forecast = parse_forecast(completion[0]["content"], len(targets))
        rewards.append(0.0 if forecast is None else 1.0 / (1.0 + mase(forecast, targets, history)))
    return rewards


def build_dataset(domain: str = DEFAULT_DOMAIN) -> Dataset:
    train_examples = load_timesx_split(domain).train
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": render_prompt(example)}],
                "past_values": list(example.past_values),
                "future_values": list(example.future_values),
            }
            for example in train_examples
        ]
    )


def main() -> None:
    dataset = build_dataset()

    config = AsyncGRPOConfig(
        output_dir="async_grpo_timesx",
        save_strategy="no",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_generations=8,
        max_completion_length=512,
        # Qwen3 reasons by default, and here it never converges -- it just goes in circles second-guessing the
        # timestamps instead of answering. Turning it off gets a real answer out most of the time instead.
        chat_template_kwargs={"enable_thinking": False},
        # num_train_epochs, not max_steps: per_device_train_batch_size counts samples not questions here, and rows
        # are packed by token count by default, so a step count wouldn't map to a known number of epochs anyway.
        num_train_epochs=2,
        learning_rate=1e-5,
        report_to="trackio",
        trackio_space_id="async-grpo-timesx",
        project="async-grpo-timesx",
        log_completions=True,
        logging_steps=1,
    )
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3.5-4B",
        args=config,
        train_dataset=dataset,
        reward_funcs=[format_reward, mase_reward],
    )
    trainer.train()


if __name__ == "__main__":
    main()
