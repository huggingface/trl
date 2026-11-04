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

Training and generation share one weight copy: the continuous batching engine decodes from the
tensors the optimizer updates, so there is no inference server and no weight sync. The engine
generates between optimizer steps, and every completion is produced by the current policy (at
most `rollouts_in_flight` rollouts behind the policy).

This recipe is the tuned configuration for the long-completion regime (Qwen3-14B, thinking
completions of 1-4k tokens, 4x B200). Each knob below is what moved the needle:

- a micro-batch is packed in one row with explicit sample boundaries, so no pad-token compute and
  flash attention's varlen kernels attend within each sample.
- `rollouts_in_flight=512`: a decode step is latency-bound, so its cost barely depends on how many
  sequences it carries (on this model at tp=4, 34 ms with 128 in flight, 44 ms with 512). A finished
  rollout is replaced immediately by the next one, and the engine wants as many as the KV cache
  holds: raise `max_memory_percent` together with this. Completions then lag the policy by
  `rollouts_in_flight / samples per step` optimizer steps (16 here), which the 14B tolerates; a 0.6B
  at the same lag learned visibly slower, so keep the lag small on small models.
- `per_device_train_batch_size=16` with `gradient_accumulation_steps=2`: fewer, larger forwards
  per step; advantages are computed at scoring time, so accumulation does not change the loss. A
  single forward of all 32 samples runs out of memory next to the KV cache.
- `learning_rate=1e-5`: the reward goes from 0.41 to 0.92 in the first 50 steps and holds there,
  with a clip ratio of 0.014; at 1e-6 it stays between 0.2 and 0.3 over 200 steps.
- `tp_size` should be the smallest that fits the model: tp=2, 4 and 8 give 2.6, 2.9 and 3.0
  samples/s, so a wider split buys almost nothing per GPU. The world size must equal `tp_size`
  until transformers' device mesh has a replicate dimension again.

Measured on 4x B200 with Qwen3-14B: 6.6 s per optimizer step of 32 samples (~45k trained tokens),
200 steps in 22 minutes.

torchrun --nproc-per-node 4 examples/zero_sync_grpo_math/zero_sync_grpo_math.py
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
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_generations=8,
        max_completion_length=4096,
        chat_template_kwargs={"enable_thinking": True},
        learning_rate=1e-5,
        rollouts_in_flight=512,
        # Fraction of free VRAM for the KV cache; the rest is for activations and gradients.
        continuous_batching_config={"max_memory_percent": 0.5},
        # The smallest split that fits a 14B
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
