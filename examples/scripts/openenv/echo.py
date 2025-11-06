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

"""
Simple script to run GRPO training with OpenEnv's Echo environment and a vLLM server. The reward function encourages
longer completions.

Setup:

```sh
uv pip install git+https://github.com/meta-pytorch/OpenEnv.git
```

Usage (2 GPUs required):

# Start the docker container for the Echo environment (recommended). Alternatively, you can run it locally or directly from a HF Space.
```sh
docker run -d -p 8001:8001 registry.hf.space/openenv-echo-env:latest
```

# Spin up server

```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
```

# Run training

```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/echo.py
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with Echo environment and vLLM.")

    parser.add_argument("--env-host", type=str, default="0.0.0.0", help="Host for the Echo environment.")
    parser.add_argument("--env-port", type=int, default=8001, help="Port for the Echo environment.")
    parser.add_argument(
        "--env-mode",
        choices=["local", "docker", "space"],
        default="docker",
        help="Where to run the Echo environment: 'local' to launch it, 'docker' if already running, or 'space' to use a remote Space URL.",
    )
    parser.add_argument(
        "--gen-url",
        type=str,
        default="http://0.0.0.0:8000/generate/",
        help="Base URL for the vLLM generation endpoint.",
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


def rollout_func(
    prompts: list[str], args: GRPOConfig, processing_class, client: EchoEnv, gen_url: str
) -> dict[str, list]:
    """Generate completions via vLLM and compute environment rewards."""
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

    response = requests.post(gen_url, json=payload)
    if response.status_code != 200:
        print(f"Error response: {response.text}")
    response.raise_for_status()

    result = response.json()
    completions_text = processing_class.batch_decode(result["completion_ids"], skip_special_tokens=True)

    env_result = client.reset()
    env_rewards = []
    for msg in completions_text:
        env_result = client.step(EchoAction(message=msg))
        env_rewards.append(env_result.reward)

    result["env_reward"] = env_rewards
    return result


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
    elif args.env_mode == "docker":
        env_url = f"http://{args.env_host}:{args.env_port}"
        server_process = None
        print(f"🌍 Using existing Echo Environment (Docker) at: {env_url}")
    elif args.env_mode == "space":
        env_url = args.env_host
        server_process = None
        print(f"🚀 Using Hugging Face Space environment at: {env_url}")
    else:
        raise ValueError(f"Unknown environment mode: {args.env_mode}")

    gen_url = args.gen_url
    client = EchoEnv(base_url=env_url)
    dataset = load_dataset(args.dataset, split="train[:1000]")

    training_args = GRPOConfig(
        output_dir=f"{args.model.split('/')[-1]}-GRPO-Rollout",
        vllm_mode="server",
        use_vllm=True,
        logging_steps=1,
        report_to="trackio",
        num_train_epochs=1,
        max_completion_length=2048,
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
        print("🛑 Terminating Echo Environment server...")
        server_process.terminate()


if __name__ == "__main__":
    main()
