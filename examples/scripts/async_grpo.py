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

"""
CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B \
    --weight-transfer-config '{"backend":"nccl"}' \
    --max-model-len 9216

LOG_LEVEL=DEBUG CUDA_VISIBLE_DEVICES=0 accelerate launch examples/scripts/async_grpo.py
"""

import logging
import os

from datasets import load_dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.rewards import accuracy_reward


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("trl").setLevel(logging.DEBUG)


def format_sample(sample):
    return {"prompt": sample["messages"][:1], "solution": sample["answer"]}


def main() -> None:
    dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train[:10000]")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    config = AsyncGRPOConfig(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_completion_length=4096,
        max_steps=10,
        report_to="trackio",
        trackio_space_id=None,
        project="async_grpo",
        log_completions=True,
    )
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-4B",
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
    )
    trainer.train()


if __name__ == "__main__":
    main()
