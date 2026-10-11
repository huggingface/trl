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
#     "peft",
#     "trackio",
# ]
# ///

"""AsyncGRPO training of a binary-market forecaster on Prophet Arena (https://www.prophetarena.co).

Each prompt is one Prophet Arena market at its earliest recorded snapshot: an event title, one named market, the
reference material available at that snapshot, a resolution criterion, and the market close time. The model outputs
a single probability that the market resolves YES; it is not shown the market price or the resolved outcome.

Reward is the Brier score `1 - (p - y)^2` for outcome `y`, a strictly proper scoring rule. A malformed completion
(no parseable probability) gets `0.0` from both reward functions below, same convention as `examples/async_grpo_timesx`.

This is a port of thinking-machines-lab/tinker-cookbook's `recipes/forecasting` (same task, same Brier reward, same
subset of the data) from Tinker's RL loop onto TRL's AsyncGRPOTrainer. Unlike that project's other recipes, this
one's output shape -- one calibrated probability per binary question -- is exactly what Prophet Arena's live
leaderboard grades, so the resulting checkpoint could be served as a real Prophet Arena agent.

Data is loaded via `datasets.load_dataset` from https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-1200
(MIT-licensed). See `data.py` for the chronological, event-level train/validation split.

CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve Qwen/Qwen3-0.6B \
    --port 8317 \
    --max-model-len 8192 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    --enable-lora --max-lora-rank 32 --max-loras 1

CUDA_VISIBLE_DEVICES=0 accelerate launch examples/async_grpo_prophet_arena/async_grpo_prophet_arena.py
"""

from __future__ import annotations

from data import load_prophet_arena_split, parse_probability, render_prompt
from datasets import Dataset
from peft import LoraConfig

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


def format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    """1.0 if the completion parses to a probability in [0, 1], else 0.0.

    Not in `reward_funcs` below -- `brier_reward` already scores malformed completions 0.0, and adding this as a
    second equal-weight reward caused the same collapse it did in `examples/async_grpo_timesx`. Kept for manual
    inspection.
    """
    return [1.0 if parse_probability(completion[0]["content"]) is not None else 0.0 for completion in completions]


def brier_reward(completions: list[list[dict[str, str]]], outcome: list[int], **kwargs) -> list[float]:
    """`1 - (p - y)^2`, or 0.0 for a malformed completion."""
    rewards = []
    for completion, y in zip(completions, outcome, strict=True):
        probability = parse_probability(completion[0]["content"])
        rewards.append(0.0 if probability is None else 1.0 - (probability - y) ** 2)
    return rewards


def build_dataset() -> Dataset:
    train_examples = load_prophet_arena_split().train
    return Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": render_prompt(example)}], "outcome": example.outcome}
            for example in train_examples
        ]
    )


def main() -> None:
    dataset = build_dataset()

    # RL loop hyperparameters mirror tinker-cookbook's recipe: 32 forecasts per question (`num_generations`) and
    # temperature 1.0. There's no exact "16 questions per optimizer step" here the way tinker-cookbook has: by
    # default `token_budget` is set to the vLLM server's `max_model_len` and rows are packed by token count
    # (`TokenBudgetBatcher`), not by `per_device_train_batch_size` samples -- see `AsyncGRPOConfig`'s docstring.
    # `gradient_accumulation_steps` still means what it always does (accumulate over that many packed micro-batches
    # before stepping) regardless of how those micro-batches were packed, so it's set the same as
    # `examples/async_grpo_math` rather than left at `1`, to avoid an effective-batch-size regression.
    # `learning_rate=1e-4` and `num_train_epochs=2` (matching their `max_steps=128`) are theirs exactly, now that
    # AsyncGRPOTrainer has `peft_config` support. Full fine-tuning at these settings reliably collapsed into a
    # repeating-token loop -- lowering the rate only delayed it, never fixed it. LoRA is the actual fix.
    config = AsyncGRPOConfig(
        output_dir="async_grpo_prophet_arena",
        save_strategy="no",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_generations=32,
        # Capped well under their 24,576 (tuned for a 27B reasoning model at high effort); Qwen3-0.6B needs nowhere
        # near that, so raise it (and `--max-model-len` in the vLLM command above) only for a larger/more
        # reasoning-heavy model.
        max_completion_length=2048,
        num_train_epochs=2,
        learning_rate=1e-4,
        temperature=1.0,
        # Default port 8000 collides with other jobs on shared Slurm nodes.
        vllm_server_base_url="http://localhost:8317",
        report_to="trackio",
        trackio_space_id="async-grpo-prophet-arena",
        project="async-grpo-prophet-arena",
        log_completions=True,
    )
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        args=config,
        train_dataset=dataset,
        reward_funcs=[brier_reward],
        # Rank 32, matching tinker-cookbook. Plain LoRA only -- no modules_to_save/DoRA/bias, nothing on
        # lm_head/embed_tokens -- or vLLM can't serve the adapter and silently falls back to merged-weight sync.
        peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"),
    )
    trainer.train()
    trainer.end()


if __name__ == "__main__":
    main()
