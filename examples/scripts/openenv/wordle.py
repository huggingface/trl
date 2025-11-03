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
GRPO training for Wordle using TRL's `GRPOTrainer` and the TextArena OpenEnv environment.

Usage:
    # First, start the TextArena Wordle server (Docker or local):
    TEXTARENA_ENV_ID=Wordle-v0 TEXTARENA_NUM_PLAYERS=1 \
        python -m src.envs.textarena_env.server.app

    # Start the vLLM server with your model
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000

    # Then run this training script:
    CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import requests
from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer


# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from envs.textarena_env import TextArenaAction, TextArenaEnv
from envs.textarena_env.models import TextArenaMessage
from envs.textarena_env.rewards import (
    extract_feedback_counts,
    extract_guess,
    extract_wordle_feedback,
)


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
        default="willcb/Qwen3-1.7B-Wordle",
        help="Model identifier passed to GRPOTrainer for fine-tuning.",
    )
    parser.add_argument(
        "--textarena-url",
        default="https://burtenshaw-textarena.hf.space",
        help="Base URL for the TextArena Wordle environment.",
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
        default=5,
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
        default=64,
        help="Gradient accumulation steps for GRPO training.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
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
        default=2,
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
        "--vllm-endpoint",
        default=os.getenv("VLLM_ENDPOINT", "http://localhost:8000/generate/"),
        help="Endpoint for the vLLM server.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=60,
        help="Timeout (in seconds) for vLLM HTTP requests.",
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
    prompt_section = prompt_text.strip() if prompt_text.strip() else "Wordle-v0"
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return (
        f"Game prompt:\n{prompt_section}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )


def request_vllm_completion(
    prompt: str,
    trainer_args: GRPOConfig,
    endpoint: str,
    timeout: int,
    fallback: argparse.Namespace,
) -> dict[str, list]:
    payload: dict[str, object] = {
        "prompts": [prompt],
        "n": 1,
        "temperature": getattr(trainer_args, "temperature", fallback.temperature),
        "max_tokens": getattr(trainer_args, "max_completion_length", fallback.max_new_tokens),
        "logprobs": True,
    }

    top_k = getattr(trainer_args, "top_k", fallback.top_k)
    if top_k is not None:
        payload["top_k"] = top_k

    top_p = getattr(trainer_args, "top_p", fallback.top_p)
    if top_p is not None:
        payload["top_p"] = top_p

    min_p = getattr(trainer_args, "min_p", None)
    if min_p is not None:
        payload["min_p"] = min_p

    repetition_penalty = getattr(trainer_args, "repetition_penalty", None)
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty

    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    prompt_ids = data.get("prompt_ids") or data.get("prompt_token_ids") or [[]]
    completion_ids = data.get("completion_ids") or data.get("completion_token_ids") or [[]]
    logprobs = data.get("logprobs") or data.get("completion_logprobs") or [[]]
    texts = data.get("completions") or data.get("completion_texts") or data.get("texts")

    return {
        "prompt_ids": prompt_ids[0] if prompt_ids else [],
        "completion_ids": completion_ids[0] if completion_ids else [],
        "logprobs": [float(lp) for lp in (logprobs[0] if logprobs else [])],
        "text": (texts[0] if texts else None),
    }


def scale_repetition_score(previous_occurrences: int, max_occurrences: int) -> float:
    """Scale the repetition score based on the number of previous occurrences from 0 to 1"""
    if max_occurrences == 0:
        return 0.0
    return (max_occurrences - previous_occurrences) / max_occurrences


def rollout_once(
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    args: GRPOConfig,
    dataset_prompt: str,
    cli_args: argparse.Namespace,
    system_prompt: str,
) -> dict[str, list]:
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    raw_rewards: list[float] = []
    green_scores: list[float] = []
    yellow_scores: list[float] = []
    repetition_scores: list[float] = []
    correct_scores: list[float] = []
    guess_counts: dict[str, int] = {}

    for _turn in range(cli_args.max_turns):
        # when the game is over the environment will return a done=True
        if result.done:
            break

        # set up the prompt for the model
        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(base_prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        # generate the completion from the model using vLLM
        vllm_result = request_vllm_completion(
            prompt_text,
            args,
            endpoint=cli_args.vllm_endpoint,
            timeout=cli_args.request_timeout,
            fallback=cli_args,
        )
        prompt_ids.extend(vllm_result["prompt_ids"])
        completion_ids.extend(vllm_result["completion_ids"])
        logprobs.extend(vllm_result["logprobs"])
        completion_text = vllm_result.get("text") or tokenizer.decode(
            vllm_result["completion_ids"], skip_special_tokens=True
        )
        # extract the guess from the completion
        guess = extract_guess(completion_text)

        # step the environment with the guess
        result = env.step(TextArenaAction(message=guess))
        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)

        # Update guess counts
        previous_occurrences = guess_counts[guess]
        repetition_score = scale_repetition_score(previous_occurrences, len(guess_counts))
        guess_counts[guess] += 1

        # calculate custom reward signals from the feedback
        if not feedback:
            green_score = 0.0
            yellow_score = 0.0
        else:
            green_count, yellow_count = extract_feedback_counts(feedback)
            green_score = green_count / 5.0
            yellow_score = yellow_count / 5.0

        repetition_scores.append(repetition_score)
        green_scores.append(green_score)
        yellow_scores.append(yellow_score)
        correct_scores.append(correct_score)

    correct_reward_value = correct_scores[-1] if correct_scores else (raw_rewards[-1] if raw_rewards else 0.0)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "raw_rewards": raw_rewards,
        "correct_reward": correct_reward_value,
        "green_reward": green_scores[-1] if green_scores else 0.0,
        "yellow_reward": yellow_scores[-1] if yellow_scores else 0.0,
        "repetition_reward": repetition_scores[-1] if repetition_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------


def rollout_func(
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    args: GRPOConfig,
    cli_args: argparse.Namespace,
    system_prompt: str,
) -> dict[str, list]:
    all_prompt_ids: list[list[int]] = []
    all_completion_ids: list[list[int]] = []
    all_logprobs: list[list[float]] = []
    correctness_rewards: list[float] = []
    green_rewards: list[float] = []
    yellow_rewards: list[float] = []
    repetition_rewards: list[float] = []
    num_generations = args.num_generations or cli_args.num_generations

    for _ in range(num_generations):
        for prompt_text in prompts:
            rollout_stats = rollout_once(
                env=env,
                tokenizer=tokenizer,
                args=args,
                dataset_prompt=prompt_text,
                cli_args=cli_args,
                system_prompt=system_prompt,
            )
            all_prompt_ids.append(rollout_stats["prompt_ids"])
            all_completion_ids.append(rollout_stats["completion_ids"])
            all_logprobs.append(rollout_stats["logprobs"])
            correctness_rewards.append(rollout_stats["correct_reward"])
            green_rewards.append(rollout_stats["green_reward"])
            yellow_rewards.append(rollout_stats["yellow_reward"])
            repetition_rewards.append(rollout_stats["repetition_reward"])

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "correct_reward": correctness_rewards,
        "green_reward": green_rewards,
        "yellow_reward": yellow_rewards,
        "repetition_reward": repetition_rewards,
    }


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def reward_correct(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("correct_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_greens(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("green_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_yellows(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("yellow_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_repetition(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("repetition_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    cli_args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(cli_args.tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    env = TextArenaEnv(base_url=cli_args.textarena_url)

    system_prompt = resolve_system_prompt(cli_args.system_prompt_path)

    dataset = Dataset.from_dict({"prompt": [cli_args.dataset_prompt] * cli_args.dataset_size})

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"wordle-grpo-{sanitize_name(cli_args.model_id)}-{timestamp}"
    output_dir = Path(cli_args.output_dir or default_output_dir)

    grpo_config = GRPOConfig(
        vllm_mode="server",
        use_vllm=True,
        output_dir=str(output_dir),
        num_train_epochs=cli_args.num_epochs,
        learning_rate=cli_args.learning_rate,
        weight_decay=cli_args.weight_decay,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        per_device_train_batch_size=cli_args.per_device_batch_size,
        warmup_steps=cli_args.warmup_steps,
        num_generations=cli_args.num_generations,
        max_completion_length=cli_args.max_new_tokens,
        logging_steps=cli_args.logging_steps,
        save_strategy="steps",
        save_steps=cli_args.save_interval,
        save_total_limit=cli_args.save_total_limit,
    )

    grpo_config.run_name = cli_args.run_name or f"run-{timestamp}"
    grpo_config.project = cli_args.project or f"group-{sanitize_name(cli_args.model_id)}"
    grpo_config.trackio_space_id = cli_args.trackio_space_id

    def wrapped_rollout(prompts: list[str], args: GRPOConfig, processing_class) -> dict[str, list]:
        return rollout_func(
            env=env,
            tokenizer=tokenizer,
            prompts=prompts,
            args=args,
            cli_args=cli_args,
            system_prompt=system_prompt,
        )

    trainer = GRPOTrainer(
        model=cli_args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correct,
            reward_greens,
            reward_yellows,
            reward_repetition,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=wrapped_rollout,
    )

    print("Starting GRPO training with Wordle environment...")
    print(f"Using {cli_args.num_generations} rollouts per dataset prompt")

    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
