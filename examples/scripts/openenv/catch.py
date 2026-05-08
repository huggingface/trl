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
#     "trl[vllm,peft]",
#     "trackio",
#     "kernels",
#     "openenv-openspiel-env @ git+https://huggingface.co/spaces/openenv/openspiel_env",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Catch environment (OpenSpiel) and vLLM. The reward function
is based on the catch game where the agent tries to catch falling balls.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/openspiel_env
```

Setup (Option B - Clone OpenEnv repo, for development):

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
import subprocess
import sys
import time
from pathlib import Path

import requests
from datasets import Dataset
from openspiel_env import OpenSpielEnv
from openspiel_env.models import OpenSpielAction

from trl import GRPOConfig, GRPOTrainer, RichProgressCallback


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

You have the following tools available:
- `move(direction)`: Move the paddle left or right. Direction must be "left" or "right".
- `stay`: Do nothing and let the ball fall one step.

Observe the grid, determine where the ball is relative to the paddle, then move accordingly.
"""


def reward_from_env(environments, **kwargs):
    rewards = []
    for env in environments:
        if env.done:
            # Catch gives +1 for catching, -1 for missing. Clamp to [0, 1] for GRPO advantage estimation.
            rewards.append(max(env.reward, 0.0))
        else:
            rewards.append(0.0)  # Incomplete episode
    return rewards


def main():
    args = parse_args()

    # Select environment mode — all modes resolve to env_url
    if args.env_mode == "local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = start_env_server(args.env_host, args.env_port)
    elif args.env_mode == "docker-local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = None
        print(f"🌍 Using existing OpenSpiel Environment (Docker) at: {env_url}")
    elif args.env_mode == "docker-image":
        _bootstrap = OpenSpielEnv.from_docker_image(args.env_image)
        env_url = _bootstrap.base_url
        server_process = None
        print("🌍 Using OpenSpiel Environment (Docker) from local Image")
    elif args.env_mode == "docker-hub":
        _bootstrap = OpenSpielEnv.from_hub(args.env_image)
        env_url = _bootstrap.base_url
        server_process = None
        print("🌍 Using existing OpenSpiel Environment (Docker) from Hub Image")
    elif args.env_mode == "space":
        env_url = args.env_host
        server_process = None
        print(f"🌍 Using Hugging Face Space environment at: {env_url}")
    else:
        raise ValueError(f"Unknown environment mode: {args.env_mode}")

    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": BASE_PROMPT}]] * args.dataset_size})

    class CatchEnv:
        ROWS = 10
        COLS = 5

        def __init__(self):
            self.client = OpenSpielEnv(base_url=env_url)
            self.reward = 0.0
            self.done = False

        @staticmethod
        def _format_obs(info_state: list[float]) -> str:
            """Convert the flat 50-float observation into a readable text description."""
            rows, cols = CatchEnv.ROWS, CatchEnv.COLS
            ball_row = ball_col = paddle_col = None
            for idx, val in enumerate(info_state):
                if val == 1.0:
                    r, c = divmod(idx, cols)
                    if r < rows - 1:
                        ball_row, ball_col = r + 1, c + 1
                    else:
                        paddle_col = c + 1
            parts = []
            if ball_row is not None and ball_col is not None:
                parts.append(f"Ball: row {ball_row}/{rows}, column {ball_col}/{cols}")
            if paddle_col is not None:
                parts.append(f"Paddle: column {paddle_col}/{cols}")
            if ball_col is not None and paddle_col is not None:
                diff = ball_col - paddle_col
                if diff < 0:
                    parts.append(f"The ball is {abs(diff)} column(s) to the LEFT of the paddle.")
                elif diff > 0:
                    parts.append(f"The ball is {diff} column(s) to the RIGHT of the paddle.")
                else:
                    parts.append("The ball is directly above the paddle.")
            return "\n".join(parts)

        def reset(self, **kwargs) -> str:
            env_result = self.client.reset()
            self.reward = 0.0
            self.done = env_result.observation.done
            return self._format_obs(env_result.observation.info_state)

        def _do_action(self, action_id: int) -> str:
            if self.done:
                raise ValueError("Episode is done.")
            env_result = self.client.step(OpenSpielAction(action_id=action_id, game_name="catch"))
            self.reward = env_result.reward or 0.0
            self.done = env_result.observation.done
            return self._format_obs(env_result.observation.info_state)

        def move(self, direction: str) -> str:
            """Move the paddle left or right.

            Args:
                direction: Direction to move, either "left" or "right".

            Returns:
                The observation after moving.
            """
            if direction == "left":
                action_id = 0
            elif direction == "right":
                action_id = 2
            else:
                raise ValueError(f"Invalid direction {direction!r}: must be 'left' or 'right'.")
            return self._do_action(action_id)

        def stay(self) -> str:
            """Do nothing and let the ball fall one step.

            Returns:
                The observation after staying.
            """
            return self._do_action(1)

    training_args = GRPOConfig(
        output_dir=f"{args.model.split('/')[-1]}-GRPO-Catch",
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=0.2,
        logging_steps=1,
        log_completions=True,
        report_to="trackio",
        trackio_space_id=f"{args.model.split('/')[-1]}-GRPO-Catch",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_generations=8,
        max_completion_length=4096,
        gradient_accumulation_steps=16,
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        args=training_args,
        train_dataset=dataset,
        environment_factory=CatchEnv,
        callbacks=[RichProgressCallback()],
    )

    try:
        trainer.train()
    finally:
        if server_process:
            print("🛑 Terminating environment server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()
