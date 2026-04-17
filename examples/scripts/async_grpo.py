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
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
# ]
# ///

"""
pip install math_verify

CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
    --max-model-len 2048 \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}'

CUDA_VISIBLE_DEVICES=0 accelerate launch examples/scripts/async_grpo.py
"""

from datasets import load_dataset

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
        output_dir="async_grpo_gsm8k",
        save_strategy="no",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        max_completion_length=1024,
        chat_template_kwargs={"enable_thinking": False},
        max_steps=200,
        learning_rate=1e-5,
        report_to="trackio",
        trackio_space_id="async-grpo-gsm8k",
        project="async-grpo-gsm8k",
        log_completions=True,
    )
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
    )
    trainer.train()


if __name__ == "__main__":
    main()
