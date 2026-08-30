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
#     "kernels",
# ]
# ///

"""
Zero-sync GRPO on GSM8K with long thinking completions, tuned for throughput.

Training and generation share one weight copy: the continuous batching engine decodes through a
view of the training model, so there is no inference server and no weight sync. The engine
generates between optimizer steps, and every completion is produced by the current policy (at
most `generation_ahead` steps stale).

This recipe is the tuned configuration for the long-completion regime (Qwen3-14B, ~3-4k-token
thinking completions, 4x H100). Each knob below is what moved the needle:

- `packed_training`: samples are packed back to back, so no pad-token compute (this also defaults
  the model to flex attention, which skips the masked-out cross-sample blocks).
- `generation_ahead=16`: a 4k rollout spans many optimizer steps, so the engine needs a deep
  lookahead to keep its decode batch full. Going 8 -> 16 -> 32 kept improving throughput at 14B
  (7.4 -> 8.5 -> 9.8% MFU) at the cost of completions up to that many steps stale; past the point
  where in-flight rollouts crowd the KV cache it regresses, so raise `max_memory_percent` before
  raising this further.
- `gradient_accumulation_steps`: splits the step into forwards that fit in memory next to the KV
  cache; advantages are computed at scoring time, so accumulation does not change the loss.
- `tp_size` should be the smallest that fits the model: decode is all-reduce latency-bound, so
  wider TP slows generation (14B at tp=8 decodes slower than at tp=4). The ranks left over hold
  replicas of the model, which train on their own prompts and sum their gradients, so more GPUs
  means more replicas rather than a wider split. On 16 GPUs, tp=4 with four replicas is 2.4x
  tp=8 with two.

Measured on 4x H100 with Qwen3-14B: ~75k trained tokens per optimizer step, ~3.1k decoded
tokens/s, 94% average SM utilization, generation and training together reaching ~10% MFU. On 8 and
16 H100 the same recipe does 4.1k and 9.2k trained tokens/s, the second being 2.2x the first.

torchrun --nproc-per-node 4 examples/zero_sync_grpo_math/zero_sync_grpo_math.py
torchrun --nproc-per-node 8 examples/zero_sync_grpo_math/zero_sync_grpo_math.py  # tp=4, two replicas
"""

import os
import re

from datasets import load_dataset

from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer


SYSTEM_PROMPT = "Solve the problem. Put your final answer after '####'."


def format_sample(sample):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
        ],
        "solution": sample["answer"].split("####")[-1].strip().replace(",", ""),
    }


def extract_answer(text):
    if "####" in text:
        match = re.search(r"-?[\d,]+\.?\d*", text.rsplit("####", 1)[1])
        if match:
            return match.group().replace(",", "").rstrip(".")
    return None


def accuracy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    return [1.0 if extract_answer(content) == gold else 0.0 for content, gold in zip(contents, solution, strict=True)]


def main():
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    config = ZeroSyncGRPOConfig(
        output_dir="zero-sync-grpo-gsm8k",
        save_strategy="no",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_completion_length=4096,
        chat_template_kwargs={"enable_thinking": True},
        generation_ahead=16,
        # Fraction of free VRAM for the KV cache; the rest is for activations and gradients.
        continuous_batching_config={"max_memory_percent": 0.25},
        packed_training=True,
        # The smallest split that fits a 14B; whatever ranks are left over hold replicas of it.
        tp_size=min(4, int(os.environ.get("WORLD_SIZE", "1"))),
        max_steps=200,
    )
    trainer = ZeroSyncGRPOTrainer(
        model="Qwen/Qwen3-14B",
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
    )
    trainer.train()


if __name__ == "__main__":
    main()
