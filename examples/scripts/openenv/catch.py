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
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from datasets import Dataset
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import OpenSpielAction

from trl import GRPOConfig, GRPOTrainer, RichProgressCallback, apply_chat_template


"""
Simple script to run GRPO training with OpenEnv's Catch environment (OpenSpiel) and a vLLM server. The reward function
is based on the catch game where the agent tries to catch falling balls.

Setup:

```sh
uv pip install git+https://github.com/meta-pytorch/OpenEnv.git
uv pip install open_spiel rich trackio
```

Usage (2 GPUs required):

# Spin up vLLM server

```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
```

# Run training

```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/catch.py
```
"""

GEN_URL = "http://0.0.0.0:8000/generate/"
ENV_URL = "http://0.0.0.0:8001"

BASE_PROMPT = """You are an AI agent playing the game **Catch**.

### Game Description
- The game is played on a **10×5 grid**.
- There is one **falling ball** and one **paddle** that you control at the bottom.
- The objective is to **move the paddle left or right to catch the ball** as it falls.
- The episode ends when the ball reaches the bottom row:
  - You get **+1 reward** if you catch it.
  - You get **–1 reward** if you miss it.

### Observation Format

- `observation`: a list of **50 numbers (floats)** representing the entire grid, flattened row by row.
  - Each cell contains `1.0` if it is occupied (either by the ball or the paddle), or `0.0` if it is empty.
  - The positions of the two `1.0` values indicate where the **ball** and **paddle** currently are.
- `legal_actions`: a list of integers representing which actions are currently allowed.

### Actions Each action is a discrete integer:
- `0` → Move paddle **left**
- `1` → **Stay** (no movement)
- `2` → Move paddle **right**

### Output Format Respond **only with one integer** representing your chosen action: `0`, `1`, or `2`.

### Current Observation
"""

# Start the OpenSpiel server in background
print("⚡ Starting FastAPI server for OpenSpiel Catch Environment...")

# Determine the correct path
work_dir = str(Path.cwd().parent.absolute())

server_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "envs.openspiel_env.server.app:app", "--host", "0.0.0.0", "--port", "8001"],
    env={**os.environ, "PYTHONPATH": f"{work_dir}/src"},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=work_dir,
)

print("⏳ Waiting for server to start...")
time.sleep(5)

# Check if server is running
try:
    response = requests.get(f"{ENV_URL}/health", timeout=2)
    print("\n✅ OpenSpiel Catch Environment server is running!")
except Exception as e:
    print(f"\n❌ Server failed to start: {e}")
    print("\n📋 Checking error output...")
    server_process.poll()
    if server_process.stderr:
        stderr = server_process.stderr.read()
        if stderr:
            print(stderr)
    raise


# Create HTTP client for OpenSpiel Catch Environment
client = OpenSpielEnv(base_url=f"{ENV_URL}")


def rollout_func(prompts: list[str], args: GRPOConfig, processing_class) -> dict[str, list]:
    """
    Custom rollout function that generates completions via vLLM server and computes environment rewards.

    The catch game expects action IDs (integers). We'll parse the model's text output to extract action choices.

    Args:
        prompts: List of prompts to generate from
        args: GRPOConfig containing all sampling parameters
        processing_class: Tokenizer/processor for decoding completions

    Returns:
        Dict containing prompt_ids, completion_ids, logprobs, and env_reward
    """
    # Run full episodes for each generation to get episode rewards
    env_rewards = []
    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []

    for base_prompt in prompts:
        for _ in range(args.num_generations):
            # Run episode: Reset environment and loop until done
            env_result = client.reset()
            obs = env_result.observation
            total_reward = 0.0

            episode_prompt_ids = []
            episode_completion_ids = []
            episode_logprobs = []

            # TODO: parallelise!
            while not obs.done:
                # FIXME: handle the addition of observation to prompt more cleanly, ideally without a train_dataset
                episode_msg = {"prompt": [{"role": "user", "content": f"{base_prompt}\n\n{obs.info_state}\n"}]}
                episode_prompt = apply_chat_template(episode_msg, processing_class)

                # Generate action from model
                gen_payload = {
                    "prompts": [episode_prompt["prompt"]],
                    "n": 1,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": -1 if args.top_k is None else args.top_k,
                    "min_p": 0.0 if args.min_p is None else args.min_p,
                    "max_tokens": args.max_completion_length,
                    "repetition_penalty": args.repetition_penalty,
                }
                gen_response = requests.post(GEN_URL, json=gen_payload)
                gen_response.raise_for_status()
                gen_result = gen_response.json()

                # Collect prompt_ids, completion_ids, and logprobs from this step
                episode_prompt_ids.extend(gen_result["prompt_ids"][0])
                episode_completion_ids.extend(gen_result["completion_ids"][0])
                episode_logprobs.extend(gen_result["logprobs"][0])

                completion_text = processing_class.batch_decode(
                    gen_result["completion_ids"], skip_special_tokens=True
                )[0]

                # Parse action from completion
                action_id = 0  # default
                numbers = re.findall(r"\b([0-2])\b", completion_text)
                if numbers:
                    action_id = int(numbers[0])
                elif obs.legal_actions:
                    action_id = obs.legal_actions[0]

                # Take action in environment
                env_result = client.step(OpenSpielAction(action_id=action_id, game_name="catch"))
                reward = env_result.reward if env_result.reward is not None else 0.0
                total_reward += reward
                obs = env_result.observation

            # Store episode results
            env_rewards.append(total_reward)
            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_completion_ids)
            all_logprobs.append(episode_logprobs)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": env_rewards,
    }


dataset = Dataset.from_dict({"prompt": [BASE_PROMPT] * 1000})


def reward_from_env(completions, **kwargs):
    """Reward function that uses the environment reward from the catch game."""
    # Extract environment rewards from kwargs (propagated via extra_fields)
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(reward) for reward in env_rewards]
    else:
        # Fallback if env_reward is not available
        return [0.0] * len(completions)


training_args = GRPOConfig(
    output_dir="Qwen2.5-0.5B-GRPO-Catch",
    vllm_mode="server",
    use_vllm=True,
    logging_steps=1,
    report_to="trackio",
    num_train_epochs=1,
    max_completion_length=4,
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

print("🛑 Terminating environment server...")
server_process.terminate()
