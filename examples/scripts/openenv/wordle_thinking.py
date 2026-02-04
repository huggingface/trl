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
#     "trackio>=0.13.0",
#     "kernels",
#     "openenv @ git+https://github.com/meta-pytorch/OpenEnv.git",
#     "openenv_core",
# ]
# ///


"""
GRPO training for Wordle with thinking mode and env_mask support.

Usage (server mode with 2 GPUs):

Terminal 1 (vLLM server on GPU 0):
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.9

Terminal 2 (Training on GPU 1):
    CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle_thinking.py \
        --env-url https://sergiopaniego-wordle.hf.space \
        --vllm-mode server \
        --vllm-server-url http://localhost:8000 \
        --gradient-accumulation-steps 4 \
        --learning-rate 1e-6 \
        --warmup-steps 10 \
        --max-new-tokens 8192

    CUDA_VISIBLE_DEVICES=1,2 accelerate launch
        --config_file examples/accelerate_configs/fsdp2_sharding_2gpu.yaml \
        examples/scripts/openenv/wordle_thinking.py \
        --env-url https://sergiopaniego-wordle-test.hf.space \
        --vllm-mode server \
        --vllm-server-url http://localhost:8000 \
        --gradient-accumulation-steps 2 \
        --learning-rate 1e-6 \
        --warmup-steps 10 \
        --max-new-tokens 8192

"""

from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# Wordle-specific helpers
# ---------------------------------------------------------------------------

_WORDLE_GUESS_PATTERN = re.compile(r"\[[A-Za-z]{5}\]")
_WORDLE_GUESS_PATTERN_ANY = re.compile(r"\[[A-Za-z]+\]")  # Any length for fallback
_THINKING_END_PATTERN = re.compile(r"</think>", re.IGNORECASE)


def extract_guess(text: str) -> str:
    """Normalize a Wordle guess string from arbitrary text.
    
    If thinking tags are present, extract guess from after </think> ONLY.
    Never extract guesses from inside the thinking block.
    """
    # Check if there's a </think> tag - extract content after it
    think_match = _THINKING_END_PATTERN.search(text)
    if think_match:
        # Get content after </think> - ONLY search here, never inside thinking
        after_think = text[think_match.end():]
        
        # First try to find a 5-letter guess
        match = _WORDLE_GUESS_PATTERN.search(after_think)
        if match:
            return match.group(0).lower()
        
        # Fallback: find any [letters] pattern after </think>
        match_any = _WORDLE_GUESS_PATTERN_ANY.search(after_think)
        if match_any:
            return match_any.group(0).lower()
        
        # Last resort: extract first 5 letters from after </think>
        cleaned = re.sub(r"[^a-z]", "", after_think.lower())
        if len(cleaned) >= 5:
            return f"[{cleaned[:5]}]"
        elif cleaned:
            return f"[{cleaned}]"
        return "[dunno]"
    
    # No thinking tags - find the last [xxxxx] pattern in the entire text
    matches = _WORDLE_GUESS_PATTERN.findall(text)
    if matches:
        return matches[-1].lower()

    # Last resort: extract first 5 letters
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if len(cleaned) >= 5:
        return f"[{cleaned[:5]}]"
    return "[dunno]"


def extract_wordle_feedback(observation) -> str:
    """Pull the latest feedback text from a Wordle observation."""
    for message in reversed(observation.messages):
        content = message.content.strip()
        if "Feedback:" in content:
            return content.split("Feedback:", 1)[-1].strip()
    return ""


def extract_feedback_counts(feedback: str) -> tuple[int, int]:
    """Return counts of green (G) and yellow (Y) markers from feedback."""
    if not feedback:
        return (0, 0)

    lines = [line.strip() for line in feedback.split("\n") if line.strip()]
    if len(lines) < 2:
        return (0, 0)

    for line in reversed(lines):
        normalized = line.replace(" ", "")
        if normalized and all(c in "GYX" for c in normalized):
            green = normalized.count("G")
            yellow = normalized.count("Y")
            return (green, yellow)

    return (0, 0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Wordle with thinking mode."
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
        "--env-url", type=str, default="https://burtenshaw-wordle.hf.space", help="URL for the environment server."
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
        default=4096,
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
        default=1,
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


def make_user_prompt(messages: Iterable[TextArenaMessage]) -> str:
    history = format_history(messages)
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return (
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )


# ---------------------------------------------------------------------------
# Rollout with env_mask support
# ---------------------------------------------------------------------------


async def rollout_once(
    trainer: GRPOTrainer,
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_turns: int,
    max_new_tokens: int = 512,
) -> dict[str, list]:
    result = await env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    env_mask: list[int] = []  # 1 for model-generated tokens, 0 for environment tokens
    model_outputs: list[str] = []
    raw_rewards: list[float] = []
    position_scores: list[float] = []
    correct_scores: list[float] = []
    guess_counts: dict[str, int] = {}  # Track repeated guesses
    repetition_penalties: list[float] = []  # Penalty for each turn
    prev_env_output_len: int = 0

    accumulated_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    # Build initial prompt
    initial_user_prompt = make_user_prompt(observation.messages)
    initial_env_output = format_history(observation.messages) if observation.messages else ""
    prev_env_output_len = len(initial_env_output)
    initial_messages = accumulated_messages + [{"role": "user", "content": initial_user_prompt}]
    initial_prompt_text = tokenizer.apply_chat_template(
        initial_messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    )
    initial_prompt_ids = tokenizer.encode(initial_prompt_text, add_special_tokens=False)
    prompt_ids.extend(initial_prompt_ids)

    for _turn in range(max_turns):
        print(f"Turn {_turn + 1} of {max_turns}")
        if result.done:
            print(f"  [Turn {_turn + 1}] Game ended (result.done=True)")
            break

        user_prompt = make_user_prompt(observation.messages)
        messages = accumulated_messages + [{"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

        rollout_outputs = generate_rollout_completions(
            trainer,
            [prompt_text],
            generation_overrides={
                "max_tokens": max_new_tokens,
            }
        )[0]

        # Add newline before model output
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        completion_ids.extend(newline_ids)
        logprobs.extend([0.0] * len(newline_ids))
        env_mask.extend([1] * len(newline_ids))

        # Add model completion tokens and logprobs
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        env_mask.extend([1] * len(rollout_outputs["completion_ids"]))
        
        # Add newline after model output
        completion_ids.extend(newline_ids)
        logprobs.extend([0.0] * len(newline_ids))
        env_mask.extend([1] * len(newline_ids))

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )
        
        # Check if thinking was completed
        has_think_open = "<think>" in completion_text.lower()
        has_think_close = "</think>" in completion_text.lower()
        
        # If thinking started but not closed, the model hit max_new_tokens mid-thought
        # End the episode early with NEGATIVE rewards to penalize truncation
        if has_think_open and not has_think_close:
            print(f"  [Turn {_turn + 1}] Generated {len(rollout_outputs['completion_ids'])} tokens, <think>=True, </think>=False - TRUNCATED, ending episode")
            # Keep the truncated tokens but end the episode
            model_outputs.append(completion_text.strip())
            # Negative rewards to penalize truncation - model should learn to close thinking
            raw_rewards.append(-0.5)
            position_scores.append(-0.5)
            correct_scores.append(-0.5)
            repetition_penalties.append(0.0)  # No repetition penalty for truncated turn
            break
        
        guess = extract_guess(completion_text)
        model_outputs.append(completion_text.strip())
        
        # Track repetitions and calculate penalty
        prev_count = guess_counts.get(guess, 0)
        guess_counts[guess] = prev_count + 1
        # Penalty: 0 for first use, -0.5 for each repetition
        repetition_penalty = -0.5 * prev_count
        repetition_penalties.append(repetition_penalty)
        
        repeat_info = f", REPEAT x{prev_count + 1}" if prev_count > 0 else ""
        print(f"  [Turn {_turn + 1}] Generated {len(rollout_outputs['completion_ids'])} tokens, <think>={has_think_open}, </think>={has_think_close}, guess={guess}{repeat_info}")

        result = await env.step(TextArenaAction(message=guess))
        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)

        # Add env output to completion_ids with env_mask=0 (excluded from loss)
        full_env_output = format_history(observation.messages) if observation.messages else ""
        new_env_output = full_env_output[prev_env_output_len:].lstrip("\n")
        prev_env_output_len = len(full_env_output)

        if new_env_output:
            env_output_tokens = tokenizer.encode(new_env_output, add_special_tokens=False)
            completion_ids.extend(env_output_tokens)
            logprobs.extend([0.0] * len(env_output_tokens))
            env_mask.extend([0] * len(env_output_tokens))
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

    # Episode summary
    total_completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    num_thinks = total_completion.lower().count("<think>")
    num_closes = total_completion.lower().count("</think>")
    print(f"[Episode End] Turns played: {len(correct_scores)}, <think> count: {num_thinks}, </think> count: {num_closes}")
    
    correct_reward_value = correct_scores[-1] if correct_scores else (raw_rewards[-1] if raw_rewards else 0.0)

    if correct_reward_value >= 1.0:
        final_position_reward = 1.0
    else:
        final_position_reward = position_scores[-1] if position_scores else 0.0

    # Sum of repetition penalties (0 for no repeats, negative for repeats)
    repetition_reward = sum(repetition_penalties) if repetition_penalties else 0.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "raw_rewards": raw_rewards,
        "correct_reward": correct_reward_value,
        "position_reward": final_position_reward,
        "repetition_reward": repetition_reward,
        "num_turns": len(correct_scores),
        "model_outputs": model_outputs,
    }


# ---------------------------------------------------------------------------
# Reward functions
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


def compute_format_reward_thinking(model_outputs: list[str]) -> float:
    """Compute format reward for thinking mode outputs."""
    if not model_outputs:
        return 0.0
    
    thinking_pattern = re.compile(
        r"^\s*<think>.*?</think>\s*\[[A-Za-z]{5}\]\s*$",
        re.DOTALL | re.IGNORECASE
    )
    
    correct_count = sum(1 for output in model_outputs if thinking_pattern.match(output))
    return correct_count / len(model_outputs)


def reward_format_strict(completions: list[str], **kwargs) -> list[float]:
    """Format reward - pre-computed in rollout_func."""
    rewards = kwargs.get("format_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_no_repetition(completions: list[str], **kwargs) -> list[float]:
    """Penalty for repeating guesses - pre-computed in rollout_func."""
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

    env_url = args.env_url

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
        max_completion_length=None,
        logging_steps=args.logging_steps,
        log_completions=True,
        report_to="wandb",
        trackio_space_id=f"wordle-grpo-{sanitize_name(args.model_id)}-{timestamp}",
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        # Memory optimizations
        vllm_gpu_memory_utilization=0.1,
        vllm_max_model_length=32768,
        vllm_importance_sampling_correction=False,
        gradient_checkpointing=False,
        #gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        optim="adamw_torch",  # adamw_bnb_8bit not compatible with FSDP2 sharding
        max_grad_norm=1.0,
        model_init_kwargs={
            "tie_word_embeddings": False,  # Required for FSDP2 compatibility
        },
    )

    grpo_config.run_name = args.run_name or f"run-{timestamp}"
    grpo_config.project = args.project or f"wordle-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    grpo_config.trackio_space_id = args.trackio_space_id

    os.environ["WANDB_PROJECT"] = f"wordle-grpo-{sanitize_name(args.model_id)}"
    os.environ["WANDB_RUN_ID"] = f"{timestamp}"

    async def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids: list[list[int]] = []
        episode_completion_ids: list[list[int]] = []
        episode_logprobs: list[list[float]] = []
        episode_env_masks: list[list[int]] = []
        correctness_rewards: list[float] = []
        position_rewards: list[float] = []
        format_rewards: list[float] = []
        repetition_rewards: list[float] = []

        # Create client inside rollout_func to ensure it's in the correct event loop
        # (asyncio.run() creates a new loop each time, breaking WebSocket connections)
        client = TextArenaEnv(base_url=env_url, message_timeout_s=300.0)
        try:
            for _ in prompts:
                episode = await rollout_once(
                    trainer=trainer,
                    env=client,
                    tokenizer=tokenizer,
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
                format_rewards.append(compute_format_reward_thinking(episode["model_outputs"]))
                repetition_rewards.append(episode["repetition_reward"])
        finally:
            await client.close()

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "env_mask": episode_env_masks,
            "correct_reward": correctness_rewards,
            "position_reward": position_rewards,
            "format_reward": format_rewards,
            "repetition_reward": repetition_rewards,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correct,
            reward_position,
            reward_format_strict,
            reward_no_repetition,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("Starting GRPO training with Wordle environment (thinking mode + env_mask)...")
    print(f"Using {args.num_generations} rollouts per dataset prompt")

    # Client is now created/closed inside rollout_func for each batch
    trainer.train()


if __name__ == "__main__":
    main()
