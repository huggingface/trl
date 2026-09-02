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
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
# ]
# ///

"""
LoRA AsyncGRPO on GSM8K with an 8B policy, on two GPUs: vLLM on GPU 0, the trainer on GPU 1.

Each weight sync publishes only the adapter (~160 MB at r=32) instead of the merged 8B checkpoint (~16 GB in bf16). That happens automatically because the server below is started with `--enable-lora`; without that flag the same script still trains, falling back to merged-weight sync with a warning.

`--max-loras 5` covers the `max_staleness + 1` adapter versions that are servable at once, and `--max-lora-rank 32` matches `r` below. The rank is a capacity bound, so any of 1, 8, 16, 32, 64, 128, 256, 320, 512 that is >= r works.

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
vllm serve Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    --enable-lora --max-lora-rank 32 --max-loras 5

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 \
    examples/async_grpo_math/async_grpo_lora_math.py
```
"""

import os

from datasets import load_dataset
from peft import LoraConfig

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.rewards import accuracy_reward


def format_sample(sample):
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


def main() -> None:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    config = AsyncGRPOConfig(
        output_dir="async_grpo_lora_gsm8k",
        save_strategy="no",
        # `VLLM_PORT` is the variable vLLM itself serves on, so exporting it once covers both sides. Handy when the
        # server and the trainer share a machine with other jobs and port 8000 is already taken.
        vllm_server_base_url=f"http://localhost:{os.environ.get('VLLM_PORT', 8000)}",
        # An 8B policy and its vLLM server share a node here, so the trainer holds the model in bf16 rather than the
        # float32 default. Keep `--dtype bfloat16` on the server to match: a precision gap between the two biases the
        # importance ratio (https://huggingface.co/papers/2510.26788).
        dtype="bfloat16",
        # LoRA trains ~1% of the parameters, so the usual GRPO learning rate is far too small.
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_completion_length=1024,
        chat_template_kwargs={"enable_thinking": False},
        max_steps=200,
        report_to="trackio",
        trackio_space_id="async-grpo-lora-gsm8k",
        project="async-grpo-lora-gsm8k",
    )
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-8B",
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
        # A plain LoRA config: no `modules_to_save`, no DoRA, no trained bias, and no adapter on `lm_head` or
        # `embed_tokens`. Each of those would make the adapter unservable by vLLM and silently drop the run back to
        # merged-weight sync.
        peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"),
    )
    trainer.train()


if __name__ == "__main__":
    main()
