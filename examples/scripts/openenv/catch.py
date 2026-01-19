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
#     "trl[vllm]",
#     "peft",
#     "trackio",
#     "kernels",
#     "openenv-openspiel-env @ git+https://huggingface.co/spaces/openenv/openspiel_env",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Catch environment (OpenSpiel) and vLLM. The reward function
is based on the catch game where the agent tries to catch falling balls.

Setup (Option A - Install from HF Space):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/openspiel_env
```

Setup (Option B - Clone OpenEnv repo):

```sh
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/openspiel_env
uv pip install -e .
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/catch.py --env-mode space --env-host https://openenv-openspiel-env.hf.space --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/catch.py --env-mode space --env-host https://openenv-openspiel-env.hf.space --vllm-mode server --vllm-server-url http://localhost:8000
```

# Option 3: Local + Colocated vLLM (1 GPU required)

# Start the environment only if using --env-mode docker-local
```sh
docker run -d -p 8001:8001 registry.hf.space/openenv-openspiel-env:latest
```

```sh
python examples/scripts/openenv/catch.py --env-mode docker-local --vllm-mode colocate
```
"""

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
from openspiel_env import OpenSpielEnv
from openspiel_env.models import OpenSpielAction

from trl import GRPOConfig, GRPOTrainer, RichProgressCallback, apply_chat_template
from trl.experimental.openenv import generate_rollout_completions


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with OpenSpiel Catch environment and vLLM.")

    # --- Environment settings ---
    parser.add_argument("--env-host", type=str, default="0.0.0.0", help="Host for the environment server.")
    parser.add_argument("--env-port", type=int, default=8001, help="Port for the environment server.")
    parser.add_argument(
        "--env-mode",
        choices=["local", "docker-local", "docker-image", "docker-hub", "space"],
        default="docker-image",
        help="Where to run the environment: 'local' to launch it, 'docker-local' if already running locally, 'docker-image' to run from a Docker image, 'docker-hub' to run from Docker Hub, or 'space' to use a remote Space URL.",
    )
    # --- Generation and model config ---
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
    parser.add_argument(
        "--env-image", type=str, default="openspiel-env:latest", help="Docker image for the OpenSpiel environment."
    )
    parser.add_argument(
        "--vllm-mode",
        choices=["colocate", "server"],
        default="colocate",
        help="vLLM execution mode: 'colocate' or 'server'.",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000",
        help="URL for the vLLM server (only used when --vllm-mode=server).",
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


def reward_from_env(completions, **kwargs):
    rewards = kwargs.get("env_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def main():
    args = parse_args()

    # Select environment mode
    if args.env_mode == "local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = start_env_server(args.env_host, args.env_port)
    elif args.env_mode == "docker-local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = None
        print(f"🌍 Using existing OpenSpiel Environment (Docker) at: {env_url}")
    elif args.env_mode == "docker-image":
        client = OpenSpielEnv.from_docker_image(args.env_image)
        server_process = None
        print("🌍 Using OpenSpiel Environment (Docker) from local Image")
    elif args.env_mode == "docker-hub":
        client = OpenSpielEnv.from_hub(args.env_image)
        server_process = None
        print("🌍 Using existing OpenSpiel Environment (Docker) from Hub Image")
    elif args.env_mode == "space":
        env_url = args.env_host
        server_process = None
        print(f"🌍 Using Hugging Face Space environment at: {env_url}")
    else:
        raise ValueError(f"Unknown environment mode: {args.env_mode}")

    if args.env_mode != "docker-hub" and args.env_mode != "docker-image":
        client = OpenSpielEnv(base_url=env_url)
    dataset = Dataset.from_dict({"prompt": [BASE_PROMPT] * args.dataset_size})

    training_args = GRPOConfig(
        output_dir=f"{args.model.split('/')[-1]}-GRPO-Catch",
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        logging_steps=1,
        report_to="trackio",
        trackio_space_id=f"{args.model.split('/')[-1]}-GRPO-Catch",
        num_train_epochs=1,
        max_completion_length=4,
        gradient_accumulation_steps=4,
    )

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        """Generate completions via vLLM (colocated or server) and compute environment rewards."""
        env_rewards: list[float] = []
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        tokenizer = trainer.processing_class

        for base_prompt in prompts:
            env_result = client.reset()
            obs = env_result.observation
            total_reward = 0.0

            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []

            while not obs.done:
                episode_msg = {"prompt": [{"role": "user", "content": f"{base_prompt}\n\n{obs.info_state}\n"}]}
                episode_prompt = apply_chat_template(episode_msg, tokenizer)
                rollout_output = generate_rollout_completions(trainer, [episode_prompt["prompt"]])[0]

                episode_prompt_ids.extend(rollout_output["prompt_ids"])
                episode_completion_ids.extend(rollout_output["completion_ids"])
                episode_logprobs.extend(rollout_output["logprobs"])

                completion_text = tokenizer.batch_decode([rollout_output["completion_ids"]], skip_special_tokens=True)[
                    0
                ]
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

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        args=training_args,
        train_dataset=dataset,
        rollout_func=rollout_func,
        callbacks=[RichProgressCallback()],
    )

    trainer.train()
    time.sleep(5)

    if server_process:
        print("🛑 Terminating environment server...")
        server_process.terminate()


if __name__ == "__main__":
    main()
