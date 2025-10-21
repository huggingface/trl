from datasets import load_dataset
import requests
from trl import GRPOConfig, GRPOTrainer
import subprocess
import time
import sys
import os
from envs.echo_env.models import (
    EchoAction,
)
from envs.echo_env import EchoEnv
from pathlib import Path
"""
Simple script to run GRPO training with OpenEnv's Echo environment and a vLLM server. The reward function encourages
longer completions.

Usage (2 GPUs required):

-- Spin up server -- CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port
8000

-- Run this script -- CUDA_VISIBLE_DEVICES=1 python trl/experimental/openenv/echo.py
"""

GEN_URL = "http://0.0.0.0:8000/generate/"
ENV_URL = "http://0.0.0.0:8001"

# Start the Echo server in background
print("⚡ Starting FastAPI server for Echo Environment...")

# Determine the correct path
work_dir = str(Path.cwd().parent.absolute())

server_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn",
     "envs.echo_env.server.app:app",
     "--host", "0.0.0.0",
     "--port", "8001"],
    env={**os.environ,
         "PYTHONPATH": f"{work_dir}/src"},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=work_dir
)

# Wait for server to start
print("⏳ Waiting for server to start...")
time.sleep(5)

# Check if server is running
import requests
try:
    response = requests.get(f'{ENV_URL}/health', timeout=2)
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
print("✅ Client created!")

def rollout_func(prompts, **sampling_kwargs):

    # Make request to TRL's custom /generate/ endpoint
    payload = {
        "prompts": prompts,
        "n": sampling_kwargs.get("n", 1),
        "temperature": sampling_kwargs.get("temperature", 1.0),
        "top_p": sampling_kwargs.get("top_p", 1.0),
        "top_k": sampling_kwargs.get("top_k", -1),
        "min_p": sampling_kwargs.get("min_p", 0.0),
        "max_tokens": sampling_kwargs.get("max_tokens", 128),
        "repetition_penalty": sampling_kwargs.get("repetition_penalty", 1.0),
    }

    print(f"Sending request to {GEN_URL}")
    response = requests.post(GEN_URL, json=payload)

    if response.status_code != 200:
        print(f"Error response: {response.text}")

    response.raise_for_status()
    result = response.json()

    # Decode for env communication
    processing_class = sampling_kwargs.get("processing_class", None)

    completions_text = processing_class.batch_decode(result["completion_ids"], skip_special_tokens=True)

    # Flush env
    env_result = client.reset()

    # Take an action (HTTP POST /step)
    print("\n📤 Calling client.step()...")

    for msg in completions_text:
        env_result = client.step(EchoAction(message=msg))

    # Get state (HTTP GET /state)
    state = client.state()
    # print(f"\n📊 Episode state:")
    # print(f"   • episode_id: {state.episode_id}")
    # print(f"   • step_count: {state.step_count}")

    # print(f"Response keys: {result.keys()}")
    # print(f"Response shapes: {[(k, len(v) if isinstance(v, list) else 'not-list') for k, v in result.items()]}")
    # print(f"=== rollout_func completed ===\n")

    return result

dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:1000]")

def reward_len(completions, **kwargs):
    """Reward function that rewards longer completions."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(content)) for content in completion_contents]

training_args = GRPOConfig(
    output_dir="scratch/Qwen2.5-0.5B-GRPO-Rollout",
    vllm_mode="server",
    use_vllm=True,
    logging_steps=1,
    report_to=["trackio", "wandb"],
    num_train_epochs=1,
    num_generations=16,
    max_completion_length=4096,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    rollout_func=rollout_func,
)
trainer.train()