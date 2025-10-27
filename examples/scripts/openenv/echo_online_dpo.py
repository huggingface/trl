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
import torch
from datasets import load_dataset
from envs.echo_env import EchoEnv
from envs.echo_env.models import EchoAction
from transformers import AutoTokenizer

from trl import OnlineDPOConfig, OnlineDPOTrainer, RichProgressCallback
from trl.models import unwrap_model_for_generation


"""
Online DPO training with OpenEnv's Echo environment using the TRAINER'S MODEL for generation.
This example shows how to use a custom rollout function that:
1. Generates completions using the trainer's model (no vLLM server needed!)
2. Computes environment rewards from OpenEnv
3. Returns both for training

Setup:

```sh
pip install git+https://github.com/meta-pytorch/OpenEnv.git
```

Usage (single GPU - everything on one device!):

```sh
python examples/scripts/openenv/echo_online_dpo_with_model.py
```
"""

ENV_URL = "http://127.0.0.1:8001"

print("⚡ Starting FastAPI server for Echo Environment...")
# Workaround if you can't run the env with Docker
work_dir = str(Path.cwd().parent.absolute())
server_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "envs.echo_env.server.app:app", "--host", "127.0.0.1", "--port", "8001"],
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


def rollout_func_with_model(prompts: list[str], trainer: OnlineDPOTrainer) -> dict:
    """
    Custom rollout function that generates completions using the trainer's model and computes environment rewards.

    This function demonstrates the NEW signature that accepts a 'trainer' parameter, allowing direct access
    to the model for generation without needing vLLM.

    Args:
        prompts: List of prompts to generate from
        trainer: The OnlineDPOTrainer instance (provides access to model, accelerator, etc.)

    Returns:
        Dict containing prompt_ids, completion_ids, and env_reward
    """
    if trainer is None:
        raise ValueError(
            "This rollout function requires the trainer parameter. "
            "Make sure you're using a version of OnlineDPOTrainer that supports this feature."
        )

    print(f"🎲 Generating completions for {len(prompts)} prompts using trainer's model...")

    device = trainer.accelerator.device

    # 1. Tokenize prompts
    processing_class = trainer.processing_class
    args = trainer.args
    prompt_inputs = processing_class(
        text=prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=False,
    )

    # Move to device
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

    # 2. Generate 2 completions per prompt using the trainer's model
    all_prompt_ids = []
    all_completion_ids = []
    all_completions_text = []

    # Unwrap model for generation (handles FSDP, DeepSpeed, etc.)
    with unwrap_model_for_generation(
        trainer.model, trainer.accelerator, gather_deepspeed3_params=args.ds3_gather_for_generation
    ) as unwrapped_model:
        unwrapped_model.eval()
        with torch.no_grad():
            for gen_idx in range(2):  # OnlineDPO requires exactly 2 completions per prompt
                print(f"  Generation {gen_idx + 1}/2...")

                # Generate
                outputs = unwrapped_model.generate(
                    **prompt_inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    top_p=args.top_p,
                    top_k=args.top_k if args.top_k is not None else 50,
                    do_sample=True if args.temperature > 0 else False,
                    pad_token_id=processing_class.pad_token_id,
                    eos_token_id=processing_class.eos_token_id,
                )

                # Extract completions (remove prompt part)
                prompt_length = prompt_inputs["input_ids"].shape[1]
                completion_ids = outputs[:, prompt_length:]

                # Decode completions
                completions_text = processing_class.batch_decode(completion_ids, skip_special_tokens=True)

                # Store results
                for i in range(len(prompts)):
                    all_prompt_ids.append(prompt_inputs["input_ids"][i].tolist())
                    all_completion_ids.append(completion_ids[i].tolist())
                    all_completions_text.append(completions_text[i])

        unwrapped_model.train()

    print(f"  ✓ Generated {len(all_completions_text)} completions")

    # 3. Step through the environment to get rewards for each completion
    print("🌍 Computing environment rewards...")
    env_result = client.reset()
    env_rewards = []
    for msg in all_completions_text:
        env_result = client.step(EchoAction(message=msg))
        env_rewards.append(env_result.reward)

    print(f"  ✓ Computed {len(env_rewards)} rewards")

    # 4. Return results in the expected format
    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "env_reward": env_rewards,  # Extra field passed to reward function
    }


def reward_from_env(completions, **kwargs):
    """Reward function that uses the environment reward from kwargs."""
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(reward) for reward in env_rewards]
    else:
        # Fallback if env_reward is not available
        return [0.0] * len(completions)


# Load dataset and tokenizer
dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train[:100]")  # Small dataset for testing
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Training configuration
training_args = OnlineDPOConfig(
    output_dir="Qwen2.5-0.5B-OnlineDPO-Echo-ModelGen",
    use_vllm=False,  # ← No vLLM! Use trainer's model instead
    logging_steps=1,
    report_to="none",
    num_train_epochs=1,
    max_new_tokens=64,  # Shorter for faster generation
    max_length=512,  # Max total sequence length
    temperature=0.7,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    bf16=True,
)

print("\n🏋️  Creating trainer...")
trainer = OnlineDPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    processing_class=tokenizer,
    reward_funcs=reward_from_env,
    args=training_args,
    train_dataset=dataset,
    rollout_func=rollout_func_with_model,  # ← Custom rollout with model access!
    callbacks=[RichProgressCallback()],
)

print("\n🚀 Starting training...")
print("=" * 80)
trainer.train()
print("=" * 80)

# Give time for background threads to finish
time.sleep(5)

print("\n🛑 Terminating Echo Environment server...")
server_process.terminate()

print("\n✅ Training complete!")
