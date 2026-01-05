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

# /// script
# dependencies = [
#     "trl[vllm]",
#     "peft",
#     "trackio>=0.13.0",
#     "kernels",
#     "openenv @ git+https://github.com/meta-pytorch/OpenEnv.git",
#     "openenv_core",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Echo environment and vLLM. The reward function encourages
longer completions.

Setup:

```sh
# uv pip install git+https://github.com/meta-pytorch/OpenEnv.git
# Hotfix: https://github.com/huggingface/trl/pull/4740
uv pip install git+https://github.com/meta-pytorch/OpenEnv.git@bf5e968286e0d49cdc03fd904d48faff4b15a437 openenv_core==0.1.1
```

Usage:

# Start the environment only if using --env-mode docker-local; In other modes, the env is automatically managed by the script.
```sh
docker run -d -p 8001:8001 registry.hf.space/openenv-echo-env:latest
```

# Option 1: Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/echo.py --vllm-mode colocate
```

# Option 2: Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/echo.py --vllm-mode server --vllm-server-url http://localhost:8000
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
from datasets import load_dataset
from envs.echo_env import EchoEnv
from envs.echo_env.models import EchoAction

from trl import GRPOConfig, GRPOTrainer, RichProgressCallback
from trl.experimental.openenv import generate_rollout_completions


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with Echo environment and vLLM.")

    parser.add_argument("--env-host", type=str, default="0.0.0.0", help="Host for the Echo environment.")
    parser.add_argument("--env-port", type=int, default=8001, help="Port for the Echo environment.")
    parser.add_argument(
        "--env-mode",
        choices=["local", "docker-local", "docker-image", "docker-hub", "space"],
        default="docker-image",
        help="Where to run the Echo environment: 'local' to launch it, 'docker-local' if already running locally, 'docker-image' to run from a Docker image, 'docker-hub' to run from Docker Hub, or 'space' to use a remote Space URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model to use for training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="trl-lib/ultrafeedback-prompt",
        help="Dataset to use for training.",
    )
    parser.add_argument(
        "--env-image", type=str, default="echo-env:latest", help="Docker image for the Echo environment."
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
    """Launch the Echo environment server locally."""
    env_url = f"http://{env_host}:{env_port}"
    print(f"⚡ Starting FastAPI server for Echo Environment on {env_url}...")

    work_dir = str(Path.cwd().parent.absolute())
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "envs.echo_env.server.app:app", "--host", env_host, "--port", str(env_port)],
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
        print("\n✅ Echo Environment server is running!")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        if process.stderr:
            print(process.stderr.read())
        raise

    return process


def reward_from_env(completions, **kwargs):
    """Extract environment rewards for training."""
    env_rewards = kwargs.get("env_reward", [])
    return [float(r) for r in env_rewards] if env_rewards else [0.0] * len(completions)


def main():
    args = parse_args()

    # Select environment mode
    if args.env_mode == "local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = start_env_server(args.env_host, args.env_port)
    elif args.env_mode == "docker-local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = None
        print(f"🌍 Using existing Echo Environment (Docker) at: {env_url}")
    elif args.env_mode == "docker-image":
        client = EchoEnv.from_docker_image(args.env_image)
        server_process = None
        print("🌍 Using Echo Environment (Docker) from local Image")
    elif args.env_mode == "docker-hub":
        client = EchoEnv.from_hub(args.env_image)
        server_process = None
        print("🌍 Using existing Echo Environment (Docker) from Hub Image")
    elif args.env_mode == "space":
        env_url = args.env_host
        server_process = None
        print(f"🌍 Using Hugging Face Space environment at: {env_url}")
    else:
        raise ValueError(f"Unknown environment mode: {args.env_mode}")

    if args.env_mode != "docker-hub" and args.env_mode != "docker-image":
        client = EchoEnv(base_url=env_url)
    dataset = load_dataset(args.dataset, split="train[:1000]")

    training_args = GRPOConfig(
        output_dir=f"{args.model.split('/')[-1]}-GRPO-Rollout",
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        logging_steps=1,
        report_to="trackio",
        trackio_space_id=f"{args.model.split('/')[-1]}-GRPO-Rollout",
        num_train_epochs=1,
        max_completion_length=2048,
        gradient_accumulation_steps=4,
    )

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        outputs = generate_rollout_completions(trainer, prompts)
        tokenizer = trainer.processing_class

        completions_text = [tokenizer.decode(output["completion_ids"], skip_special_tokens=True) for output in outputs]

        env_result = client.reset()
        env_rewards: list[float] = []
        for message in completions_text:
            env_result = client.step(EchoAction(message=message))
            env_rewards.append(env_result.reward)

        return {
            "prompt_ids": [output["prompt_ids"] for output in outputs],
            "completion_ids": [output["completion_ids"] for output in outputs],
            "logprobs": [output["logprobs"] for output in outputs],
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
        print("🛑 Terminating Echo Environment server...")
        server_process.terminate()


if __name__ == "__main__":
    main()
