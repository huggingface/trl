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
#     "openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env",
# ]
# ///

"""
Simple script to run GRPO training with OpenEnv's BrowserGym environment and vLLM for LLMs.

This script is optimized for text-only Language Models (LLMs). It uses the accessibility
tree text from BrowserGym, making it memory-efficient.

The environment runs on a Hugging Face Space by default.

Setup (Option A - Install from HF Space):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/browsergym_env
```

Setup (Option B - Clone OpenEnv repo):

```sh
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/browsergym_env
uv pip install -e .
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/browsergym_llm.py --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8001
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/browsergym_llm.py --vllm-mode server --vllm-server-url http://localhost:8001
```
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from browsergym_env import BrowserGymAction, BrowserGymEnv
from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training for BrowserGym MiniWoB using OpenEnv environment.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-0.6B",
        help="Model identifier passed to GRPOTrainer for fine-tuning.",
    )
    parser.add_argument(
        "--space-url",
        type=str,
        default="https://openenv-browsergym-env.hf.space",
        help="URL for the Hugging Face Space running the BrowserGym environment.",
    )
    parser.add_argument(
        "--benchmark",
        default="miniwob",
        help="BrowserGym benchmark to use (miniwob, webarena, etc.).",
    )
    parser.add_argument(
        "--task-name",
        default="click-test",
        help="Specific task within the benchmark (e.g., click-test, click-button).",
    )
    parser.add_argument(
        "--dataset-prompt",
        default="Complete the web task successfully.",
        help="Prompt text used to seed the training dataset.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of entries to include in the synthetic training dataset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of new tokens to request from vLLM for each action.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used during rollout generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter forwarded to vLLM.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top-p sampling parameter forwarded to vLLM.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for GRPO training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay applied during optimization.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Gradient accumulation steps for GRPO training.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps for the scheduler.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of rollout generations per dataset prompt.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Interval (in steps) between checkpoint saves.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where training outputs and checkpoints are stored.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name for logging systems.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional project identifier for logging systems.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help="vLLM execution mode: 'colocate' or 'server'.",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8001",
        help="URL for the vLLM server (only used when --vllm-mode=server).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Frequency of logging steps for GRPO training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debugging output during rollouts.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You control a web browser through BrowserGym actions.
You must complete the given web task by interacting with the page.

Available actions:
- noop() - Do nothing
- click(bid) - Click element with BrowserGym ID (the number in brackets)
- fill(bid, text) - Fill input field with text
- send_keys(text) - Send keyboard input
- scroll(direction) - Scroll up/down

The page structure shows elements as: [bid] element_type 'element_text'
For example: [13] button 'Click Me!' means bid='13'

Reply with exactly ONE action on a single line, e.g.:
click('13')
fill('42', 'hello world')
noop()

Do not include explanations or multiple actions."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_user_prompt(goal: str, step_num: int, axtree: str, error: str = "") -> str:
    """Create user prompt from observation."""
    prompt_parts = [f"Step {step_num + 1}"]

    if goal:
        prompt_parts.append(f"Goal: {goal}")

    if error:
        prompt_parts.append(f"Previous action error: {error}")

    # Include accessibility tree (truncated for context)
    if axtree:
        max_len = 2000
        axtree_truncated = axtree[:max_len] + "..." if len(axtree) > max_len else axtree
        prompt_parts.append(f"Page structure:\n{axtree_truncated}")

    prompt_parts.append("What action do you take?")

    return "\n\n".join(prompt_parts)


def parse_action(response_text: str) -> str:
    """Parse BrowserGym action from model response."""
    # Extract first line that looks like an action
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "(" in line and ")" in line:
            return line

    # Fallback to noop if no valid action found
    return "noop()"


def rollout_once(
    trainer: GRPOTrainer,
    env: BrowserGymEnv,
    tokenizer: AutoTokenizer,
    dataset_prompt: str,
    max_steps: int,
    debug: bool = False,
) -> dict[str, list]:
    """Run one episode and collect training data (text-only, no screenshots)."""
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    completion_rewards: list[float] = []

    for step_num in range(max_steps):
        if result.done:
            break

        # Create prompt from observation (text-only using accessibility tree)
        goal = observation.goal or dataset_prompt
        axtree = observation.axtree_txt or ""
        error = observation.error if observation.last_action_error else ""

        user_prompt = make_user_prompt(goal, step_num, axtree, error)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Generate action with vLLM
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Parse and execute action
        action_str = parse_action(completion_text)

        if debug:
            print(f"Step {step_num + 1}: {action_str}")

        # Take action in environment
        result = env.step(BrowserGymAction(action_str=action_str))
        observation = result.observation

        # Track rewards
        step_reward = float(result.reward or 0.0)
        step_rewards.append(step_reward)

        # Reward shaping: success is most important
        if result.done and step_reward > 0:
            completion_rewards.append(1.0)  # Task completed successfully
        elif result.done and step_reward == 0:
            completion_rewards.append(0.0)  # Task failed
        else:
            completion_rewards.append(step_reward)  # Intermediate reward

    # Final reward is based on task completion
    final_reward = completion_rewards[-1] if completion_rewards else 0.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "step_rewards": step_rewards,
        "completion_reward": final_reward,
    }


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def reward_completion(completions: list[str], **kwargs) -> list[float]:
    """Reward for task completion."""
    rewards = kwargs.get("completion_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Connect to BrowserGym environment via Hugging Face Space
    client = BrowserGymEnv(base_url=args.space_url)
    print(f"🌍 Using Hugging Face Space environment at: {args.space_url}")

    dataset = Dataset.from_dict({"prompt": [args.dataset_prompt] * args.dataset_size})

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"browsergym-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=0.4,
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,  # Must be divisible by num_generations
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        report_to="trackio",
        trackio_space_id=f"browsergym-grpo-{sanitize_name(args.model_id)}-{timestamp}",
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    grpo_config.run_name = args.run_name or f"run-{timestamp}"
    grpo_config.project = args.project or f"group-{sanitize_name(args.model_id)}"

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        completion_rewards: list[float] = []

        if args.debug:
            print(f"\n[DEBUG] rollout_func called with {len(prompts)} prompts (LLM mode, text-only)")

        for i, prompt_text in enumerate(prompts):
            if args.debug:
                print(f"[DEBUG] Processing prompt {i + 1}/{len(prompts)}")
            episode = rollout_once(
                trainer=trainer,
                env=client,
                tokenizer=trainer.processing_class,
                dataset_prompt=prompt_text,
                max_steps=args.max_steps,
                debug=args.debug,
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            completion_rewards.append(episode["completion_reward"])

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "completion_reward": completion_rewards,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[reward_completion],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("=" * 80)
    print("Starting GRPO training with BrowserGym environment (LLM mode)")
    print(f"Benchmark: {args.benchmark}")
    print(f"Task: {args.task_name}")
    print(f"Model: {args.model_id}")
    print("Mode: LLM (text-only, using accessibility tree)")
    print(f"Using {args.num_generations} rollouts per dataset prompt")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    try:
        trainer.train()
        print("\nTraining completed successfully!")
    finally:
        client.close()


if __name__ == "__main__":
    main()
