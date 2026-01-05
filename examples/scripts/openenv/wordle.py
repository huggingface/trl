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
Simple script to run GRPO training with OpenEnv's Wordle environment and vLLM.

Setup:

```sh
# uv pip install git+https://github.com/meta-pytorch/OpenEnv.git
# Hotfix: https://github.com/huggingface/trl/pull/4740
uv pip install git+https://github.com/meta-pytorch/OpenEnv.git@bf5e968286e0d49cdc03fd904d48faff4b15a437 openenv_core==0.1.1
```

Usage:

# Start the environment only if using --env-mode docker-local; In other modes, the env is automatically managed by the script.
```sh
docker run -d -p 8001:8001 registry.hf.space/burtenshaw-textarena:latest
# or TEXTARENA_ENV_ID=Wordle-v0 TEXTARENA_NUM_PLAYERS=1 python -m src.envs.textarena_env.server.app
```

# Option 1: Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/wordle.py --vllm-mode colocate
```

# Option 2: Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle.py --vllm-mode server --vllm-server-url http://localhost:8000
```
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions


# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from envs.textarena_env import TextArenaAction, TextArenaEnv
from envs.textarena_env.models import TextArenaMessage
from envs.textarena_env.rewards import extract_feedback_counts, extract_guess, extract_wordle_feedback


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
    parser.add_argument("--env-host", type=str, default="0.0.0.0", help="Host for the environment server.")
    parser.add_argument("--env-port", type=int, default=8001, help="Port for the environment server.")
    parser.add_argument(
        "--env-mode",
        choices=["docker-local", "docker-image", "docker-hub", "space"],
        default="docker-image",
        help="Where to run the environment: 'docker-local' if already running locally, 'docker-image' to run from a Docker image, 'docker-hub' to run from Docker Hub, or 'space' to use a remote Space URL.",
    )
    parser.add_argument(
        "--env-image", type=str, default="textarena-env:latest", help="Docker image for the TextArena environment."
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


def scale_repetition_score(previous_occurrences: int, max_occurrences: int) -> float:
    """Scale the repetition score based on the number of previous occurrences from 0 to 1"""
    if max_occurrences == 0:
        return 0.0
    return (max_occurrences - previous_occurrences) / max_occurrences


def rollout_once(
    trainer: GRPOTrainer,
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int,
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
    guess_counts: defaultdict[str, int] = defaultdict(int)

    for _turn in range(max_turns):
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

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
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
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Select environment mode
    if args.env_mode == "docker-local":
        env_url = f"http://{args.env_host}:{args.env_port}"
        client = TextArenaEnv(base_url=env_url)
        print(f"🌍 Using existing TextArena Environment (Docker) at: {env_url}")
    elif args.env_mode == "docker-image":
        client = TextArenaEnv.from_docker_image(args.env_image)
        print("🌍 Using TextArena Environment (Docker) from local Image")
    elif args.env_mode == "docker-hub":
        client = TextArenaEnv.from_hub(args.env_image)
        print("🌍 Using existing TextArena Environment (Docker) from Hub Image")
    elif args.env_mode == "space":
        env_url = args.env_host
        client = TextArenaEnv(base_url=env_url)
        print(f"🌍 Using Hugging Face Space environment at: {env_url}")
    else:
        raise ValueError(f"Unknown environment mode: {args.env_mode}")

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
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        report_to="trackio",
        trackio_space_id=f"wordle-grpo-{sanitize_name(args.model_id)}-{timestamp}",
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    grpo_config.run_name = args.run_name or f"run-{timestamp}"
    grpo_config.project = args.project or f"group-{sanitize_name(args.model_id)}"
    grpo_config.trackio_space_id = args.trackio_space_id

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        correctness_rewards: list[float] = []
        green_rewards: list[float] = []
        yellow_rewards: list[float] = []
        repetition_rewards: list[float] = []

        for prompt_text in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=client,
                tokenizer=tokenizer,
                dataset_prompt=prompt_text,
                system_prompt=system_prompt,
                max_turns=args.max_turns,
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            correctness_rewards.append(episode["correct_reward"])
            green_rewards.append(episode["green_reward"])
            yellow_rewards.append(episode["yellow_reward"])
            repetition_rewards.append(episode["repetition_reward"])

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "correct_reward": correctness_rewards,
            "green_reward": green_rewards,
            "yellow_reward": yellow_rewards,
            "repetition_reward": repetition_rewards,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correct,
            reward_greens,
            reward_yellows,
            reward_repetition,
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
