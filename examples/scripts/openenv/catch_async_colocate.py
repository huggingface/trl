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
"""
Async version of Catch game training with GRPO using vLLM colocate mode.

This version demonstrates:
- vLLM colocate mode (no separate server needed!)
- Async vLLM generation for non-blocking inference
- Synchronous environment interaction (identical to catch.py)

Key design:
- Only vLLM generation is async (using AsyncVLLMColocateWrapper)
- Environment calls (client.reset(), client.step()) are synchronous
- Episodes run sequentially (environment server limitation)

Run:
```bash
# Terminal 1: Start OpenSpiel environment server
python 3rd_party/OpenEnv/examples/openspiel_api.py

# Terminal 2: Run training with async colocate
python examples/scripts/openenv/catch_async_colocate.py
```
"""

import asyncio
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
from trl.extras.vllm_colocate_async import AsyncVLLMColocateWrapper


ENV_URL = "http://0.0.0.0:8001"

# Create HTTP client for OpenSpiel Catch Environment (shared globally like catch.py)
client = OpenSpielEnv(base_url=ENV_URL)


async def play_episode_async(
    base_prompt: str,
    async_vllm: AsyncVLLMColocateWrapper,
    sampling_params,
    processing_class,
) -> dict:
    """
    Play one full episode of Catch game asynchronously.

    Args:
        base_prompt: Task description for the agent
        async_vllm: Async wrapper around vLLM LLM instance
        sampling_params: vLLM sampling parameters
        processing_class: Tokenizer/processor

    Returns:
        Dict with episode data: prompt_ids, completion_ids, logprobs, total_reward
    """
    # Use global client (like catch.py)
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

        # ASYNC VLLM GENERATION (runs in thread pool, non-blocking!)
        outputs = await async_vllm.generate_async(
            prompts=[episode_prompt["prompt"]],
            sampling_params=sampling_params,
        )

        # Extract tokens and logprobs from vLLM output
        output = outputs[0]
        prompt_ids = output.prompt_token_ids
        completion_ids = output.outputs[0].token_ids
        logprobs = [token.logprob for token in output.outputs[0].logprobs] if output.outputs[0].logprobs else []

        # Collect trajectory data
        episode_prompt_ids.extend(prompt_ids)
        episode_completion_ids.extend(completion_ids)
        episode_logprobs.extend(logprobs)

        # Decode completion to get action
        completion_text = processing_class.decode(completion_ids, skip_special_tokens=True)

        # Parse action from text
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

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "total_reward": total_reward,
    }


def rollout_func_async_colocate(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
    """
    Async rollout function using vLLM colocate mode.

    Key advantages:
    - No HTTP overhead (vLLM colocated in process)
    - Async vLLM generation (non-blocking)
    - Synchronous environment interaction (identical to catch.py)

    Note: This requires vLLM colocate mode (use_vllm=True, vllm_mode="colocate")
    """
    args = trainer.args
    processing_class = trainer.processing_class

    # Get vLLM engine from trainer
    if not hasattr(trainer, "llm") or trainer.llm is None:
        raise ValueError(
            "This rollout function requires vLLM colocate mode.\n"
            "Set: use_vllm=True, vllm_mode='colocate' in GRPOConfig"
        )

    # Wrap vLLM instance for async usage
    async_vllm = AsyncVLLMColocateWrapper(trainer.llm)

    # Prepare sampling params for vLLM
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1 if args.top_k is None else args.top_k,
        min_p=0.0 if args.min_p is None else args.min_p,
        max_tokens=args.max_completion_length,
        repetition_penalty=args.repetition_penalty,
    )

    # Run full episodes for each generation to get episode rewards (like catch.py)
    env_rewards = []
    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []

    loop = asyncio.get_event_loop()
    for base_prompt in prompts:
        for _ in range(args.num_generations):
            # Run episode (only vLLM generation is async, environment interaction is synchronous)
            result = loop.run_until_complete(
                play_episode_async(
                    base_prompt,
                    async_vllm,
                    sampling_params,
                    processing_class,
                )
            )
            env_rewards.append(result["total_reward"])
            all_prompt_ids.append(result["prompt_ids"])
            all_completion_ids.append(result["completion_ids"])
            all_logprobs.append(result["logprobs"])

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": env_rewards,
    }


def reward_from_env(completions, **kwargs):
    """Reward function that uses the environment reward from the catch game."""
    # Extract environment rewards from kwargs (propagated via extra_fields from rollout_func)
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(reward) for reward in env_rewards]
    else:
        # Fallback to 0.0 if no env_reward provided
        return [0.0] * len(completions)


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


if __name__ == "__main__":
    # Training configuration with vLLM COLOCATE mode
    training_args = GRPOConfig(
        output_dir="grpo-catch-async-colocate-test",
        num_train_epochs=1,
        max_prompt_length=512,
        max_completion_length=128,
        num_generations=4,
        generation_batch_size=4,
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        bf16=True,
        # !! KEY SETTINGS FOR COLOCATE MODE !!
        use_vllm=True,
        vllm_mode="colocate",  # Run vLLM in-process
    )

    # Prepare dataset - simple task description
    dataset = Dataset.from_dict(
        {
            "prompt": [
                "You are playing the Catch game. Choose an action (0=left, 1=stay, 2=right) to catch the falling ball."
            ]
            * 100
        }
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=reward_from_env,
        args=training_args,
        train_dataset=dataset,
        rollout_func=rollout_func_async_colocate,  # ← Async colocate version!
        callbacks=[RichProgressCallback()],
    )
    trainer.train()

    # Give time for background threads to finish
    time.sleep(5)

    print("🛑 Terminating environment server...")
    server_process.terminate()
