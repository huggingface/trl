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

# ruff: noqa: T201
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset
from envs.echo_env import EchoEnv
from envs.echo_env.models import EchoAction

from trl import GRPOConfig, GRPOTrainer, RichProgressCallback


"""
Simple script to run GRPO training with OpenEnv's Echo environment and a vLLM server. The reward function encourages
longer completions.

Setup:

```sh
uv pip install git+https://github.com/meta-pytorch/OpenEnv.git
```

Usage (2 GPUs required):

# Spin up server

```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
```

# Run training

```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/echo.py
```
"""

GEN_URL = "http://0.0.0.0:8000/generate/"
ENV_URL = "http://0.0.0.0:8001"

print("⚡ Starting FastAPI server for Echo Environment...")
# Workaround if you can't run the env with Docker
work_dir = str(Path.cwd().parent.absolute())
server_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "envs.echo_env.server.app:app", "--host", "0.0.0.0", "--port", "8001"],
    env={**os.environ, "PYTHONPATH": f"{work_dir}/src"},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=work_dir,
)

print("⏳ Waiting for server to start...")
time.sleep(5)

try:
    response = requests.get(f"{ENV_URL}/health", timeout=2)
    print("\n✅ Echo Environment server is running!")
except Exception as e:
    print(f"\n❌ Server failed to start: {e}")
    print("\n📋 Checking error output...")
    server_process.poll()
    if server_process.stderr:
        stderr = server_process.stderr.read()
        if stderr:
            print(stderr)
    raise


# Create HTTP client for Echo Environment
client = EchoEnv(base_url=f"{ENV_URL}")


def rollout_func(prompts: list[str], args: GRPOConfig, processing_class) -> dict[str, list]:
    """
    Custom rollout function that generates completions via vLLM server and computes environment rewards.

    Args:
        prompts: List of prompts to generate from
        args: GRPOConfig containing all sampling parameters
        processing_class: Tokenizer/processor for decoding completions

    Returns:
        Dict containing prompt_ids, completion_ids, logprobs, and env_reward
    """
    # 1. Generate completions via vLLM inference server (running on port 8000)
    payload = {
        "prompts": prompts,
        "n": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": -1 if args.top_k is None else args.top_k,
        "min_p": 0.0 if args.min_p is None else args.min_p,
        "max_tokens": args.max_completion_length,
        "repetition_penalty": args.repetition_penalty,
    }
    response = requests.post(GEN_URL, json=payload)

    if response.status_code != 200:
        print(f"Error response: {response.text}")

    response.raise_for_status()
    result = response.json()

    completions_text = processing_class.batch_decode(result["completion_ids"], skip_special_tokens=True)

    # 2. Step through the environment to get rewards
    env_result = client.reset()
    env_rewards = []
    for msg in completions_text:
        env_result = client.step(EchoAction(message=msg))
        env_rewards.append(env_result.reward)

    # 3. Add environment rewards as extra field
    result["env_reward"] = env_rewards

    return result


def reward_from_env(completions, **kwargs):
    """Reward function that uses the environment reward."""
    # Extract environment rewards from kwargs (propagated via extra_fields)
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(reward) for reward in env_rewards]
    else:
        # Fallback if env_reward is not available
        return [0.0] * len(completions)


dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:1000]")

training_args = GRPOConfig(
    output_dir="Qwen2.5-0.5B-GRPO-Rollout",
    vllm_mode="server",
    use_vllm=True,
    logging_steps=1,
    report_to="trackio",
    num_train_epochs=1,
    max_completion_length=2048,
    gradient_accumulation_steps=4,
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_from_env,
    args=training_args,
    train_dataset=dataset,
    rollout_func=rollout_func,
    callbacks=[RichProgressCallback()],
)
trainer.train()

# Give time for background threads to finish
time.sleep(5)

print("🛑 Terminating Echo Environment server...")
server_process.terminate()
