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
#     "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Wordle environment and vLLM.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/wordle
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/wordle.py --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle.py --vllm-mode server --vllm-server-url http://localhost:8000
```

# Option 3: Local Environment + Colocated vLLM (1 GPU required)

To run the Wordle environment locally, you have several options:

## Option 3a: Using Docker Image (Recommended)

First, build the Docker image from the textarena_env directory:
```sh
cd 3rd_party/OpenEnv/envs/textarena_env
docker build -t textarena-env:latest -f server/Dockerfile .
```

Then run the environment server:
```sh
docker run -d -p 8001:8001 textarena-env:latest
```

Finally, run training pointing to local server:
```sh
python examples/scripts/openenv/wordle.py --vllm-mode colocate --env-url http://localhost:8001
```

## Option 3b: Running Server Directly

From the textarena_env directory:
```sh
cd 3rd_party/OpenEnv/envs/textarena_env
uv venv && source .venv/bin/activate
uv pip install -e .
python -m uvicorn server.app:app --reload --port 8001
```

Then in another terminal, run training:
```sh
python examples/scripts/openenv/wordle.py --vllm-mode colocate --env-url http://localhost:8001
```

## Option 3c: Using Pre-built HF Space Image

```sh
docker run -d -p 8001:8001 registry.hf.space/burtenshaw-wordle:latest
python examples/scripts/openenv/wordle.py --vllm-mode colocate --env-url http://localhost:8001
```
"""

import argparse
import re
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions


# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from textarena_env import TextArenaAction, TextArenaEnv
from textarena_env.models import TextArenaMessage
from textarena_env.rewards import extract_feedback_counts, extract_guess, extract_wordle_feedback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Wordle using the TextArena OpenEnv environment."
    )
    parser.add_argument(
        "--tokenizer-id",
        default="Qwen/Qwen3-1.7B",
        help="Model identifier used to load the tokenizer.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-1.7B",
        help="Model identifier passed to GRPOTrainer for fine-tuning.",
    )
    parser.add_argument(
        "--env-url", type=str, default="https://sergiopaniego-wordle.hf.space", help="URL for the environment server."
    )
    parser.add_argument(
        "--system-prompt-path",
        default="wordle_prompt.txt",
        help="Path to the file containing the system prompt.",
    )
    parser.add_argument(
        "--dataset-prompt",
        default="Play Wordle like an expert.",
        help="Prompt text used to seed the training dataset.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=3000,
        help="Number of entries to include in the synthetic training dataset.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Maximum number of turns to play in the Wordle environment per episode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum number of new tokens to request from vLLM for each guess.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature used during rollout generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
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
        default=1e-6,
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
        default=64,
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
        default=10,
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
        "--trackio-space-id",
        default="Wordle-GRPO",
        help="TrackIO space identifier.",
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
        default="http://localhost:8000",
        help="URL for the vLLM server (only used when --vllm-mode=server).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Frequency of logging steps for GRPO training.",
    )
    return parser.parse_args()


def resolve_system_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        prompt_path = Path(__file__).parent / path
    return prompt_path.read_text()


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_history(messages: Iterable[TextArenaMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)


def make_user_prompt(prompt_text: str, messages: Iterable[TextArenaMessage]) -> str:
    history = format_history(messages)
    # Only use messages for conversation history - the prompt is already included as the first message
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return f"Conversation so far:\n{history_section}\n\nReply with your next guess enclosed in square brackets."


def rollout_once(
    trainer: GRPOTrainer,
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int,
    max_new_tokens: int = 16,
) -> dict[str, list]:
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    env_mask: list[int] = []  # 1 for model-generated tokens, 0 for environment tokens
    model_outputs: list[str] = []
    raw_rewards: list[float] = []
    position_scores: list[float] = []
    correct_scores: list[float] = []
    prev_env_output_len: int = 0  # Track length to only add NEW portion each turn

    accumulated_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    # Build initial prompt (only once, at the start)
    # The initial env messages are included in the prompt, not completion
    base_prompt = observation.prompt or dataset_prompt
    initial_user_prompt = make_user_prompt(base_prompt, observation.messages)
    # Track initial env output length so we don't add it again
    initial_env_output = format_history(observation.messages) if observation.messages else ""
    prev_env_output_len = len(initial_env_output)
    initial_messages = accumulated_messages + [{"role": "user", "content": initial_user_prompt}]
    initial_prompt_text = tokenizer.apply_chat_template(
        initial_messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    # Tokenize initial prompt once - this is the base prompt for the entire episode.
    # GRPO expects one prompt-completion pair per episode, where:
    # - prompt_ids = the initial/base prompt (what the model sees at episode start)
    # - completion_ids = all model responses + env feedback from all turns concatenated
    # Note: The actual prompts used for generation in each turn are longer (include conversation history),
    # but we only count the initial prompt tokens here.
    initial_prompt_ids = tokenizer.encode(initial_prompt_text, add_special_tokens=False)
    prompt_ids.extend(initial_prompt_ids)

    for _turn in range(max_turns):
        if result.done:
            break

        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(base_prompt, observation.messages)
        messages = accumulated_messages + [{"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        rollout_outputs = generate_rollout_completions(
            trainer, [prompt_text], generation_overrides={"max_tokens": max_new_tokens}
        )[0]
        # Add model-generated completion tokens and logprobs with newlines for readability
        newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
        completion_ids.extend(newline_tokens)  # newline before guess
        logprobs.extend([0.0] * len(newline_tokens))
        env_mask.extend([1] * len(newline_tokens))  # newlines are part of model output format

        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        env_mask.extend([1] * len(rollout_outputs["completion_ids"]))  # model-generated tokens

        completion_ids.extend(newline_tokens)  # newline after guess
        logprobs.extend([0.0] * len(newline_tokens))
        env_mask.extend([1] * len(newline_tokens))  # newlines are part of model output format
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )
        guess = extract_guess(completion_text)
        model_outputs.append(completion_text.strip())  # Store raw model output for format reward

        result = env.step(TextArenaAction(message=guess))

        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)

        full_env_output = format_history(observation.messages) if observation.messages else ""
        new_env_output = full_env_output[prev_env_output_len:].lstrip("\n")
        prev_env_output_len = len(full_env_output)

        if new_env_output:
            env_output_tokens = tokenizer.encode(new_env_output, add_special_tokens=False)
            completion_ids.extend(env_output_tokens)  # Add to completion_ids
            logprobs.extend([0.0] * len(env_output_tokens))  # Placeholder (ignored via env_mask=0)
            env_mask.extend([0] * len(env_output_tokens))  # Environment tokens - mask out from loss
            completion_with_env = completion_text + "\n" + new_env_output
        else:
            completion_with_env = completion_text

        accumulated_messages.append({"role": "user", "content": user_prompt})
        accumulated_messages.append({"role": "assistant", "content": completion_with_env})

        if not feedback:
            position_score = 0.0
        else:
            green_count, yellow_count = extract_feedback_counts(feedback)
            position_score = (green_count + 0.5 * yellow_count) / 5.0

        position_scores.append(position_score)
        correct_scores.append(correct_score)

    # Use the final correct reward (win/lose is binary at end)
    correct_reward_value = correct_scores[-1] if correct_scores else (raw_rewards[-1] if raw_rewards else 0.0)

    # Position reward as shaping signal:
    # - If model WINS: position_reward = 1.0 (no penalty for winning fast)
    # - If model LOSES: position_reward = last attempt (where it ended up)
    if correct_reward_value >= 1.0:
        final_position_reward = 1.0
    else:
        final_position_reward = position_scores[-1] if position_scores else 0.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "raw_rewards": raw_rewards,
        "correct_reward": correct_reward_value,
        "position_reward": final_position_reward,
        "model_outputs": model_outputs,
    }


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def reward_correct(completions: list[str], **kwargs) -> list[float]:
    """Reward from environment (correct answer)."""
    rewards = kwargs.get("correct_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_position(completions: list[str], **kwargs) -> list[float]:
    """Position reward: green worth 1.0, yellow worth 0.5, normalized by 5."""
    rewards = kwargs.get("position_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def compute_format_reward(model_outputs: list[str]) -> float:
    """Compute format reward from a list of model outputs (one per turn).

    Each output should be exactly [5 letters] with optional whitespace.
    Returns proportion of correctly formatted outputs.
    """
    if not model_outputs:
        return 0.0

    exact_pattern = re.compile(r"^\s*\[[A-Za-z]{5}\]\s*$")
    correct_count = sum(1 for output in model_outputs if exact_pattern.match(output))

    return correct_count / len(model_outputs)


def reward_format_strict(completions: list[str], **kwargs) -> list[float]:
    """Format reward - pre-computed in rollout_func."""
    rewards = kwargs.get("format_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    client = TextArenaEnv(base_url=args.env_url)

    system_prompt = resolve_system_prompt(args.system_prompt_path)

    dataset = Dataset.from_dict({"prompt": [args.dataset_prompt] * args.dataset_size})

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"wordle-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        max_completion_length=1024,  # Full episode length, not per-turn
        logging_steps=args.logging_steps,
        log_completions=True,
        report_to="trackio",
        trackio_space_id=f"wordle-grpo-{sanitize_name(args.model_id)}-{timestamp}",
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        vllm_gpu_memory_utilization=0.25,
        vllm_max_model_length=8192,
        vllm_importance_sampling_mode="token_truncate",  # Less aggressive than default sequence_mask
        optim="adamw_torch",
        max_grad_norm=1.0,  # Clip gradients to prevent explosion
    )

    grpo_config.run_name = args.run_name or f"run-{timestamp}"
    grpo_config.project = args.project or f"wordle-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    grpo_config.trackio_space_id = args.trackio_space_id

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        episode_env_masks: list[list[int]] = []
        correctness_rewards: list[float] = []
        position_rewards: list[float] = []
        format_rewards: list[float] = []

        for prompt_text in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=client,
                tokenizer=tokenizer,
                dataset_prompt=prompt_text,
                system_prompt=system_prompt,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            episode_env_masks.append(episode["env_mask"])
            correctness_rewards.append(episode["correct_reward"])
            position_rewards.append(episode["position_reward"])
            format_rewards.append(compute_format_reward(episode["model_outputs"]))

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "env_mask": episode_env_masks,
            "correct_reward": correctness_rewards,
            "position_reward": position_rewards,
            "format_reward": format_rewards,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correct,
            reward_position,
            reward_format_strict,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("Starting GRPO training with Wordle environment...")
    print(f"Using {args.num_generations} rollouts per dataset prompt")

    try:
        trainer.train()
    finally:
        client.close()


if __name__ == "__main__":
    main()
