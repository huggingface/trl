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
import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with OpenSpiel Catch environment and vLLM.")

    # --- Environment settings ---
    parser.add_argument("--env-host", type=str, default="0.0.0.0", help="Host for the environment server.")
    parser.add_argument("--env-port", type=int, default=8001, help="Port for the environment server.")
    parser.add_argument(
        "--env-mode",
        choices=["local", "docker", "space"],
        default="local",
        help="Where to run the environment: 'local', 'docker', or 'space'.",
    )
    parser.add_argument(
        "--space-url",
        type=str,
        default="https://sergiopaniego-openspiel-env.hf.space",
        help="URL of the Hugging Face Space if using --env-mode space.",
    )

    # --- Generation and model config ---
    parser.add_argument(
        "--gen-url",
        type=str,
        default="http://0.0.0.0:8000/generate/",
        help="vLLM generation endpoint URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of prompts to use for training dataset.",
    )

    return parser.parse_args()


def start_env_server(env_host: str, env_port: int):
    """Launch the OpenSpiel Catch environment locally via uvicorn."""
    env_url = f"http://{env_host}:{env_port}"
    print(f"⚡ Starting FastAPI server for OpenSpiel Catch Environment on {env_url}...")

    work_dir = str(Path.cwd().parent.absolute())
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "envs.openspiel_env.server.app:app",
            "--host",
            env_host,
            "--port",
            str(env_port),
        ],
        env={**os.environ, "PYTHONPATH": f"{work_dir}/src"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=work_dir,
    )

    print("⏳ Waiting for server to start...")
    time.sleep(5)

    try:
        requests.get(f"{env_url}/health", timeout=2)
        print("\n✅ OpenSpiel Catch Environment server is running!")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        if process.stderr:
            print(process.stderr.read())
        raise

    return process


BASE_PROMPT = """You are an AI agent playing the game **Catch**.

### Game Description
- The game is played on a **10×5 grid**.
- There is one **falling ball** and one **paddle** that you control at the bottom.
- The objective is to **move the paddle left or right to catch the ball** as it falls.
- The episode ends when the ball reaches the bottom row:
  - You get **+1 reward** if you catch it.
  - You get **–1 reward** if you miss it.

### Observation Format
Each observation is a flattened 10x5 grid (list of 50 floats).
- 1.0 → occupied (ball or paddle)
- 0.0 → empty cell

### Actions:
- `0` → Move left
- `1` → Stay
- `2` → Move right

Respond **only** with one integer: `0`, `1`, or `2`.

### Current Observation
"""


def rollout_func(
    prompts: list[str], args: GRPOConfig, processing_class, client: OpenSpielEnv, gen_url: str
) -> dict[str, list]:
    """Generate completions via vLLM and compute environment rewards."""
    env_rewards = []
    all_prompt_ids, all_completion_ids, all_logprobs = [], [], []

    for base_prompt in prompts:
        for _ in range(args.num_generations):
            env_result = client.reset()
            obs = env_result.observation
            total_reward = 0.0

            episode_prompt_ids, episode_completion_ids, episode_logprobs = [], [], []

            while not obs.done:
                episode_msg = {"prompt": [{"role": "user", "content": f"{base_prompt}\n\n{obs.info_state}\n"}]}
                episode_prompt = apply_chat_template(episode_msg, processing_class)

                payload = {
                    "prompts": [episode_prompt["prompt"]],
                    "n": 1,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_completion_length,
                }
                response = requests.post(gen_url, json=payload)
                response.raise_for_status()
                result = response.json()

                episode_prompt_ids.extend(result["prompt_ids"][0])
                episode_completion_ids.extend(result["completion_ids"][0])
                episode_logprobs.extend(result["logprobs"][0])

                completion_text = processing_class.batch_decode(result["completion_ids"], skip_special_tokens=True)[0]

                numbers = re.findall(r"\b([0-2])\b", completion_text)
                action_id = int(numbers[0]) if numbers else obs.legal_actions[0]

                env_result = client.step(OpenSpielAction(action_id=action_id, game_name="catch"))
                total_reward += env_result.reward or 0.0
                obs = env_result.observation

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


def reward_from_env(completions, **kwargs):
    rewards = kwargs.get("env_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def main():
    args = parse_args()

    # Select environment mode
    if args.env_mode == "local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = start_env_server(args.env_host, args.env_port)
    elif args.env_mode == "docker":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = None
        print(f"🌍 Using existing Docker environment at {env_url}")
    elif args.env_mode == "space":
        env_url = args.space_url
        server_process = None
        print(f"🚀 Using Hugging Face Space environment at {env_url}")
    else:
        raise ValueError(f"Unknown env mode: {args.env_mode}")

    gen_url = args.gen_url
    client = OpenSpielEnv(base_url=env_url)
    dataset = Dataset.from_dict({"prompt": [BASE_PROMPT] * args.dataset_size})

    training_args = GRPOConfig(
        output_dir=f"{args.model.split('/')[-1]}-GRPO-Catch",
        vllm_mode="server",
        use_vllm=True,
        logging_steps=1,
        report_to="trackio",
        num_train_epochs=1,
        max_completion_length=4,
        gradient_accumulation_steps=4,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        args=training_args,
        train_dataset=dataset,
        rollout_func=lambda p, a, pc: rollout_func(p, a, pc, client, gen_url),
        callbacks=[RichProgressCallback()],
    )

    trainer.train()
    time.sleep(5)

    if server_process:
        print("🛑 Terminating environment server...")
        server_process.terminate()


if __name__ == "__main__":
    main()
