# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import os

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer



dataset = load_dataset("trl-lib/tldr", split="train[:1%]")

checkpoint_dir = os.path.join("/sgl-workspace/ryang/trl", "checkpoints/sgl")
os.makedirs(checkpoint_dir, exist_ok=True)


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


training_args = GRPOConfig(
    output_dir=os.path.join(checkpoint_dir, "Qwen2.5_output"),
    logging_steps=10,
    # report_to="wandb",
    # use_vllm=True,
    use_sglang=True,
    sglang_device="cuda:1",
    sglang_gpu_memory_utilization=0.9,
    sglang_server_url="http://127.0.0.1:30000",
)


trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

training_args.checkpoint_path = checkpoint_dir  # Set the checkpoint path for later use


trainer.train()
