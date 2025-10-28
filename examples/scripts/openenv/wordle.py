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
    
    # Then run this training script:
    python grpo.py
"""

from __future__ import annotations

from datetime import datetime
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import os
import requests
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from envs.textarena_env import TextArenaAction, TextArenaEnv
from envs.textarena_env.models import TextArenaMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_ID = "Qwen/Qwen3-1.7B"
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/generate/")
MAX_TURNS = 5
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.8
TOP_K = 10
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0
GRADIENT_ACCUMULATION_STEPS = 64
WARMUP_STEPS = 20
PER_DEVICE_BATCH_SIZE = 1
NUM_GENERATIONS = 2
NUM_EPOCHS = 1
DATASET_SIZE = 3000
SAVE_INTERVAL = 10
OUTPUT_DIR = f"outputs/wordle-grpo-{MODEL_ID.replace('/', '-')}-{NOW}"
RUN_ID = f"run-{NOW}"
PROJECT_ID = f"group-{MODEL_ID.replace('/', '-')}"
SPACE_ID = "Wordle-GRPO"

with open("wordle_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

DEBUG = False
if DEBUG:
    print("=" * 100)
    print("DEBUG mode enabled")
    print("=" * 100)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

env = TextArenaEnv(base_url="https://burtenshaw-textarena.hf.space")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_history(messages: Iterable[TextArenaMessage]) -> str:
    lines: List[str] = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)


def extract_guess(text: str) -> str:
    match = re.search(r"\[[A-Za-z]{5}\]", text)
    if match:
        return match.group(0).lower()

    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if len(cleaned) >= 5:
        return f"[{cleaned[:5]}]"
    return "[dunno]"


def make_user_prompt(prompt_text: str, messages: Iterable[TextArenaMessage]) -> str:
    history = format_history(messages)
    prompt_section = prompt_text.strip() if prompt_text.strip() else "Wordle-v0"
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return (
        f"Game prompt:\n{prompt_section}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )


def extract_wordle_feedback(observation: TextArenaObservation) -> str:
    for message in reversed(observation.messages):
        content = message.content.strip()
        if "Feedback:" in content:
            return content.split("Feedback:", 1)[-1].strip()
    return ""


def compute_feedback_score(feedback: str) -> float:
    latest_line = feedback.split("\n\n")[-2].strip()
    greens = latest_line.count("G") * 2
    yellows = latest_line.count("Y")
    base_score = greens + yellows
    return base_score / 10


def repeated_guess_penalty(guess: str, feedback: str) -> float:
    guess_clean = guess.strip("[]").upper()
    occurrences = len(
        re.findall(rf"\b{re.escape(guess_clean)}\b", feedback.replace(" ", ""))
    )
    return 0.0 if occurrences > 1 else 1.0


def request_vllm_completion(prompt: str, args: GRPOConfig) -> Dict[str, List]:
    payload: Dict[str, object] = {
        "prompts": [prompt],
        "n": 1,
        "temperature": getattr(args, "temperature", TEMPERATURE),
        "max_tokens": getattr(args, "max_completion_length", MAX_NEW_TOKENS),
        "logprobs": True,
    }

    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        payload["top_k"] = top_k

    top_p = getattr(args, "top_p", None)
    if top_p is not None:
        payload["top_p"] = top_p

    response = requests.post(VLLM_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    prompt_ids = data.get("prompt_ids") or data.get("prompt_token_ids") or [[]]
    completion_ids = (
        data.get("completion_ids") or data.get("completion_token_ids") or [[]]
    )
    logprobs = data.get("logprobs") or data.get("completion_logprobs") or [[]]
    texts = data.get("completions") or data.get("completion_texts") or data.get("texts")

    return {
        "prompt_ids": prompt_ids[0] if prompt_ids else [],
        "completion_ids": completion_ids[0] if completion_ids else [],
        "logprobs": [float(lp) for lp in (logprobs[0] if logprobs else [])],
        "text": (texts[0] if texts else None),
    }


def rollout_once(
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    args: GRPOConfig,
    dataset_prompt: str,
) -> Dict[str, List]:
    result = env.reset()
    observation = result.observation

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    logprobs: List[float] = []
    raw_rewards: List[float] = []
    coverage_rewards: List[float] = []
    repetition_rewards: List[float] = []

    for _turn in range(MAX_TURNS):
        if result.done:
            break
        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(base_prompt, observation.messages)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        vllm_result = request_vllm_completion(prompt_text, args)

        prompt_ids.extend(vllm_result["prompt_ids"])
        completion_ids.extend(vllm_result["completion_ids"])
        logprobs.extend(vllm_result["logprobs"])

        completion_text = vllm_result.get("text") or tokenizer.decode(
            vllm_result["completion_ids"], skip_special_tokens=True
        )
        guess = extract_guess(completion_text)

        result = env.step(TextArenaAction(message=guess))
        raw_rewards.append(float(result.reward or 0.0))

        observation = result.observation
        feedback = extract_wordle_feedback(observation)
        if not feedback:
            repetition_reward = 0.0
            coverage_reward = 0.0
        else:
            repetition_reward = repeated_guess_penalty(guess, feedback)
            coverage_reward = compute_feedback_score(feedback)
        repetition_rewards.append(repetition_reward)
        coverage_rewards.append(coverage_reward)
        if DEBUG:
            # RL IS HARRRRRRRRD
            print("=" * 100)
            print(f"Guess: {guess}")
            print(f"Feedback: {feedback}")
            print(f"Repetition reward: {repetition_reward}")
            print(f"Coverage reward: {coverage_reward}")
            print(f"Raw reward: {result.reward}")
            print("=" * 100)

    solved = bool(result.done and raw_rewards and raw_rewards[-1] > 0.0)
    correctness_reward = 1.0 if solved else 0.0
    coverage_reward = coverage_rewards[-1] if coverage_rewards else 0.0
    repetition_reward = sum(repetition_rewards) / max(1, len(repetition_rewards))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "raw_rewards": raw_rewards,
        "correct_reward": correctness_reward,
        "coverage_reward": coverage_reward,
        "repetition_reward": repetition_reward,
    }

# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------

def rollout_func(
    prompts: List[str], args: GRPOConfig, processing_class
) -> Dict[str, List]:

    all_prompt_ids: List[List[int]] = []
    all_completion_ids: List[List[int]] = []
    all_logprobs: List[List[float]] = []
    correctness_rewards: List[float] = []
    coverage_rewards: List[float] = []
    repetition_rewards: List[float] = []
    num_generations = args.num_generations or NUM_GENERATIONS

    for _ in range(num_generations):
        for prompt_text in prompts:
            rollout_stats = rollout_once(env, processing_class, args, prompt_text)
            all_prompt_ids.append(rollout_stats["prompt_ids"])
            all_completion_ids.append(rollout_stats["completion_ids"])
            all_logprobs.append(rollout_stats["logprobs"])
            correctness_rewards.append(rollout_stats["correct_reward"])
            coverage_rewards.append(rollout_stats["coverage_reward"])
            repetition_rewards.append(rollout_stats["repetition_reward"])

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "correct_reward": correctness_rewards,
        "coverage_reward": coverage_rewards,
        "repetition_reward": repetition_rewards,
    }

# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def reward_correct(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = kwargs.get("correct_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_coverage(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = kwargs.get("coverage_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_repetition(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = kwargs.get("repetition_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:

    train_dataset = Dataset.from_dict(
        {"prompt": ["Play Wordle like an expert."] * DATASET_SIZE}
    )

    grpo_config = GRPOConfig(
        vllm_mode="server",
        use_vllm=True,
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        logging_steps=1,
        save_strategy="steps",
        save_steps=SAVE_INTERVAL,
        save_total_limit=None,
    )

    grpo_config.run_name = RUN_ID
    grpo_config.project = PROJECT_ID
    grpo_config.trackio_space_id = SPACE_ID

    MODEL_ID = "willcb/Qwen3-1.7B-Wordle"

    trainer = GRPOTrainer(
        model=MODEL_ID,
        processing_class=tokenizer,
        reward_funcs=[reward_correct, reward_coverage, reward_repetition],
        train_dataset=train_dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("Starting GRPO training with Wordle environment...")
    print(f"Using {NUM_GENERATIONS} rollouts per dataset prompt")

    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()