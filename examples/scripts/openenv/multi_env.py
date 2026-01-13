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
Multi-environment, multi-reward GRPO training with Wordle and Sudoku.

This script demonstrates how to:
- Generate GRPO rollouts on multiple OpenEnv environments (Wordle + Sudoku)
- Compute multiple reward components per environment
- Use `None`-returning reward functions for multi-task training (TRL uses nansum/nanmean)

Setup:

```sh
uv pip install git+https://huggingface.co/spaces/burtenshaw/wordle
uv pip install git+https://huggingface.co/spaces/openenv/sudoku
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/multi_env.py --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/multi_env.py --vllm-mode server --vllm-server-url http://localhost:8000
```

# Full example with all flags:
```sh
python examples/scripts/openenv/multi_env.py \
    --vllm-mode colocate \
    --num-wordle 500 \
    --num-sudoku 500 \
    --num-generations 2 \
    --wordle-max-turns 6 \
    --sudoku-max-turns 100 \
    --sudoku-difficulty easy
```
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions


# Ensure src/ is on the path for TextArena imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from textarena_env import TextArenaAction, TextArenaEnv
from textarena_env.models import TextArenaMessage
from textarena_env.rewards import extract_feedback_counts, extract_guess, extract_wordle_feedback


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-environment GRPO training with Wordle and Sudoku."
    )

    # Model
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B", help="Model identifier for fine-tuning.")
    parser.add_argument("--tokenizer-id", default=None, help="Tokenizer identifier (defaults to model-id).")

    # Environment URLs
    parser.add_argument(
        "--wordle-env-url",
        type=str,
        default="https://burtenshaw-wordle.hf.space",
        help="URL for the Wordle environment server.",
    )
    parser.add_argument(
        "--sudoku-env-url",
        type=str,
        default="https://openenv-sudoku.hf.space",
        help="URL for the Sudoku environment server.",
    )

    # Dataset mixture
    parser.add_argument("--num-wordle", type=int, default=500, help="Number of Wordle prompts in the dataset.")
    parser.add_argument("--num-sudoku", type=int, default=500, help="Number of Sudoku prompts in the dataset.")

    # Prompts
    parser.add_argument("--wordle-system-prompt-path", default="wordle_prompt.txt", help="Path to Wordle system prompt.")
    parser.add_argument("--sudoku-system-prompt-path", default="sudoku_prompt.txt", help="Path to Sudoku system prompt.")
    parser.add_argument("--wordle-dataset-prompt", default="Play Wordle like an expert.", help="Wordle dataset prompt.")
    parser.add_argument("--sudoku-dataset-prompt", default="Play Sudoku like an expert.", help="Sudoku dataset prompt.")

    # Per-env rollout controls
    parser.add_argument("--wordle-max-turns", type=int, default=6, help="Maximum turns for Wordle episodes.")
    parser.add_argument("--sudoku-max-turns", type=int, default=100, help="Maximum turns for Sudoku episodes.")
    parser.add_argument(
        "--sudoku-difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="hard",
        help="Sudoku difficulty: easy=guaranteed+options, medium=only options, hard=no hints.",
    )
    parser.add_argument("--api-delay", type=float, default=0.0, help="Delay between API calls to avoid rate limiting.")

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling parameter.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling parameter.")

    # Training
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64, help="Gradient accumulation steps.")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps.")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--num-generations", type=int, default=2, help="Number of rollout generations per prompt.")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs.")

    # Checkpoints
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N steps.")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Maximum checkpoints to keep.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")

    # Logging
    parser.add_argument("--run-name", default=None, help="Run name for logging.")
    parser.add_argument("--project", default=None, help="Project name for logging.")
    parser.add_argument("--trackio-space-id", default="MultiEnv-GRPO", help="TrackIO space identifier.")
    parser.add_argument("--logging-steps", type=int, default=1, help="Logging frequency.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable verbose debugging output.")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing.")

    # vLLM
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate", help="vLLM execution mode.")
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000", help="vLLM server URL.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2, help="vLLM GPU memory utilization.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def resolve_system_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        prompt_path = Path(__file__).parent / path
    return prompt_path.read_text()


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


# ---------------------------------------------------------------------------
# Wordle helpers
# ---------------------------------------------------------------------------


def format_wordle_history(messages: Iterable[TextArenaMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)


def make_wordle_user_prompt(prompt_text: str, messages: Iterable[TextArenaMessage]) -> str:
    history = format_wordle_history(messages)
    prompt_section = prompt_text.strip() if prompt_text.strip() else "Wordle-v0"
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return (
        f"Game prompt:\n{prompt_section}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )


def scale_repetition_score(previous_occurrences: int, max_occurrences: int) -> float:
    if max_occurrences == 0:
        return 0.0
    return (max_occurrences - previous_occurrences) / max_occurrences


def wordle_rollout_once(
    trainer: GRPOTrainer,
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int,
    api_delay: float = 0.0,
) -> dict[str, Any]:
    result = env.reset()
    time.sleep(api_delay)
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
        if result.done:
            break

        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_wordle_user_prompt(base_prompt, observation.messages)
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
        guess = extract_guess(completion_text)

        result = env.step(TextArenaAction(message=guess))
        time.sleep(api_delay)
        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)

        previous_occurrences = guess_counts[guess]
        repetition_score = scale_repetition_score(previous_occurrences, len(guess_counts))
        guess_counts[guess] += 1

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
        "wordle_correct_reward": correct_reward_value,
        "wordle_green_reward": green_scores[-1] if green_scores else 0.0,
        "wordle_yellow_reward": yellow_scores[-1] if yellow_scores else 0.0,
        "wordle_repetition_reward": repetition_scores[-1] if repetition_scores else 0.0,
    }


# ---------------------------------------------------------------------------
# Sudoku helpers
# ---------------------------------------------------------------------------


def extract_sudoku_move(text: str) -> str:
    match = re.search(r"\[(\d)\s+(\d)\s+(\d)\]", text)
    if match:
        row, col, num = match.groups()
        return f"[{row} {col} {num}]"
    match = re.search(r"\[(\d)(\d)(\d)\]", text)
    if match:
        row, col, num = match.groups()
        return f"[{row} {col} {num}]"
    return ""


def is_valid_board_state(board_str: str) -> bool:
    return "R1" in board_str and "R9" in board_str and "|" in board_str


def parse_board(board_str: str) -> list[list[int]]:
    grid = [[0] * 9 for _ in range(9)]
    if not is_valid_board_state(board_str):
        return grid
    for line in board_str.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped[0] == "R" and len(line_stripped) > 1 and line_stripped[1].isdigit():
            row = int(line_stripped[1]) - 1
            cell_part = line_stripped[2:]
            col = 0
            for char in cell_part:
                if char == ".":
                    grid[row][col] = 0
                    col += 1
                elif char.isdigit():
                    grid[row][col] = int(char)
                    col += 1
    return grid


def count_filled_cells(board_str: str) -> int:
    if not is_valid_board_state(board_str):
        return 0
    grid = parse_board(board_str)
    return sum(1 for row in grid for cell in row if cell != 0)


def get_valid_numbers(grid: list[list[int]], row: int, col: int) -> set[int]:
    if grid[row][col] != 0:
        return set()
    used = set()
    for c in range(9):
        if grid[row][c] != 0:
            used.add(grid[row][c])
    for r in range(9):
        if grid[r][col] != 0:
            used.add(grid[r][col])
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] != 0:
                used.add(grid[r][c])
    return set(range(1, 10)) - used


def extract_empty_cells_with_candidates(
    board_str: str, sort_by_difficulty: bool = True
) -> list[tuple[int, int, set[int]]]:
    grid = parse_board(board_str)
    cells_with_candidates = []
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                candidates = get_valid_numbers(grid, row, col)
                cells_with_candidates.append((row + 1, col + 1, candidates))
    if sort_by_difficulty:
        cells_with_candidates.sort(key=lambda x: len(x[2]))
    return cells_with_candidates


def extract_empty_cells(board_str: str) -> list[tuple[int, int]]:
    empty_cells = []
    if not is_valid_board_state(board_str):
        return empty_cells
    for line in board_str.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped[0] == "R" and len(line_stripped) > 1 and line_stripped[1].isdigit():
            row = int(line_stripped[1])
            cell_part = line_stripped[2:]
            col = 0
            for char in cell_part:
                if char == ".":
                    col += 1
                    empty_cells.append((row, col))
                elif char.isdigit():
                    col += 1
    return empty_cells


def extract_board_only(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    board_lines = []
    in_board = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("C1") or (
            stripped and stripped[0] == "R" and len(stripped) > 1 and stripped[1].isdigit()
        ):
            in_board = True
        if in_board and (stripped.startswith("-") or stripped.startswith("R") or stripped.startswith("C1")):
            board_lines.append(line)
        elif (
            in_board
            and stripped
            and not stripped.startswith("-")
            and not (stripped[0] == "R" and len(stripped) > 1 and stripped[1].isdigit())
        ):
            break
    return "\n".join(board_lines) if board_lines else ""


def make_sudoku_compact_prompt(
    board: str,
    step: int,
    successful_moves: list[str],
    failed_moves: list[str],
    difficulty: str = "hard",
) -> str:
    cells_filled = len(successful_moves)
    summary = f"Step {step}. Progress: {cells_filled} cells filled."
    board_only = extract_board_only(board) if board else "No board available."

    tried_moves_hint = ""
    all_tried = successful_moves + failed_moves
    if all_tried:
        tried_moves_hint = f"\n\n⚠️ MOVES ALREADY TRIED (do not repeat): {', '.join(all_tried)}"

    hints = ""
    if difficulty == "easy" and board:
        cells_with_candidates = extract_empty_cells_with_candidates(board, sort_by_difficulty=True)
        if cells_with_candidates:
            guaranteed = []
            other_hints = []
            for row, col, candidates in cells_with_candidates[:10]:
                if len(candidates) == 1:
                    num = list(candidates)[0]
                    guaranteed.append(f"[{row} {col} {num}]")
                elif len(candidates) <= 3:
                    nums = ",".join(str(n) for n in sorted(candidates))
                    other_hints.append(f"({row},{col})→{nums}")
            if guaranteed:
                hints = f"\n\n🎯 GUARANTEED MOVES: {', '.join(guaranteed[:5])}"
            if other_hints:
                hints += f"\nOther options: {' | '.join(other_hints[:5])}"
    elif difficulty == "medium" and board:
        cells_with_candidates = extract_empty_cells_with_candidates(board, sort_by_difficulty=False)
        if cells_with_candidates:
            cell_hints = []
            for row, col, candidates in cells_with_candidates[:10]:
                nums = ",".join(str(n) for n in sorted(candidates))
                cell_hints.append(f"({row},{col})→{nums}")
            if cell_hints:
                hints = f"\n\nEmpty cells: {' | '.join(cell_hints)}"

    return f"{summary}\n\nBoard:\n{board_only}{tried_moves_hint}{hints}\n\nYour move:"


def check_move_targets_empty_cell(move: str, board_str: str) -> bool:
    if not move or not board_str:
        return False
    match = re.search(r"\[(\d)\s+(\d)\s+(\d)\]", move)
    if not match:
        return False
    row, col = int(match.group(1)), int(match.group(2))
    empty_cells = extract_empty_cells(board_str)
    return (row, col) in empty_cells


def extract_sudoku_feedback(observation) -> dict:
    feedback = {"valid_move": True, "got_warning": False, "board_state": ""}
    if not observation or not observation.messages:
        return feedback
    for message in observation.messages:
        content = message.content.lower() if message.content else ""
        if any(kw in content for kw in ["invalid", "error", "cannot", "already", "violation", "lost"]):
            feedback["valid_move"] = False
            if "please resubmit" in content or "avoid penalties" in content:
                feedback["got_warning"] = True
        if message.content and "|" in message.content and "R1" in message.content:
            feedback["board_state"] = message.content
    return feedback


def sudoku_rollout_once(
    trainer: GRPOTrainer,
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_turns: int,
    difficulty: str = "hard",
    api_delay: float = 0.0,
    debug: bool = False,
) -> dict[str, Any]:
    result = env.reset()
    time.sleep(api_delay)
    observation = result.observation

    last_turn_data: dict | None = None
    valid_move_scores: list[float] = []
    empty_cell_scores: list[float] = []
    correct_scores: list[float] = []
    repetition_scores: list[float] = []
    move_counts: defaultdict[str, int] = defaultdict(int)
    successful_moves: list[str] = []
    failed_moves: list[str] = []

    last_board_state = ""
    initial_filled = 0
    for message in observation.messages:
        if message.content and is_valid_board_state(message.content):
            last_board_state = message.content
            initial_filled = count_filled_cells(last_board_state)
            break

    max_filled = initial_filled

    for turn in range(max_turns):
        if result.done:
            break

        user_prompt = make_sudoku_compact_prompt(
            board=last_board_state,
            step=turn + 1,
            successful_moves=successful_moves,
            failed_moves=failed_moves,
            difficulty=difficulty,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        if debug:
            print(f"\n{'=' * 60}")
            print(f"SUDOKU STEP {turn + 1}")
            print(f"{'=' * 60}")
            print(f"USER PROMPT:\n{user_prompt}")

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        last_turn_data = {
            "prompt_ids": rollout_outputs["prompt_ids"],
            "completion_ids": rollout_outputs["completion_ids"],
            "logprobs": rollout_outputs["logprobs"],
        }

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )
        move = extract_sudoku_move(completion_text)

        if debug:
            print(f"MODEL OUTPUT: {completion_text}")
            print(f"EXTRACTED MOVE: {move}")

        result = env.step(TextArenaAction(message=move))
        time.sleep(api_delay)
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_sudoku_feedback(observation)

        if last_board_state and move:
            targets_empty = check_move_targets_empty_cell(move, last_board_state)
            empty_cell_score = 1.0 if targets_empty else -1.0
        else:
            empty_cell_score = 0.0

        is_new_move = move_counts[move] == 0
        repetition_count = move_counts[move]
        move_counts[move] += 1

        if repetition_count > 0:
            repetition_score = -min(2 ** (repetition_count - 1), 10.0)
        else:
            repetition_score = 0.0

        if not debug:
            print(f"Sudoku Step {turn + 1}: {move}")

        if feedback["valid_move"] and is_new_move:
            valid_move_score = 1.0
            if move:
                successful_moves.append(move)
        elif feedback["got_warning"]:
            valid_move_score = -0.5
            if move:
                failed_moves.append(move)
        else:
            valid_move_score = 0.0

        if feedback["board_state"] and is_valid_board_state(feedback["board_state"]):
            last_board_state = feedback["board_state"]
            current_filled = count_filled_cells(last_board_state)
            if current_filled > max_filled:
                max_filled = current_filled

        valid_move_scores.append(valid_move_score)
        empty_cell_scores.append(empty_cell_score)
        correct_scores.append(correct_score)
        repetition_scores.append(repetition_score)

    correct_reward = correct_scores[-1] if correct_scores else 0.0
    valid_move_reward = sum(valid_move_scores) / len(valid_move_scores) if valid_move_scores else 0.0
    empty_cell_reward = sum(empty_cell_scores) / len(empty_cell_scores) if empty_cell_scores else 0.0
    repetition_reward = sum(repetition_scores) / len(repetition_scores) if repetition_scores else 0.0

    remaining_to_fill = 81 - initial_filled
    if remaining_to_fill > 0:
        progress_reward = (max_filled - initial_filled) / remaining_to_fill
    else:
        progress_reward = 1.0

    if last_turn_data:
        prompt_ids = last_turn_data["prompt_ids"]
        completion_ids = last_turn_data["completion_ids"]
        logprobs_list = last_turn_data["logprobs"]
    else:
        prompt_ids = []
        completion_ids = []
        logprobs_list = []

    cells_filled = max_filled - initial_filled
    print(
        f"Sudoku Episode: empty_cell={empty_cell_reward:.2f}, valid={valid_move_reward:.2f}, "
        f"repetition={repetition_reward:.2f}, progress={progress_reward:.2f} ({cells_filled} cells), "
        f"correct={correct_reward:.2f}"
    )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs_list,
        "sudoku_correct_reward": correct_reward,
        "sudoku_valid_move_reward": valid_move_reward,
        "sudoku_empty_cell_reward": empty_cell_reward,
        "sudoku_repetition_reward": repetition_reward,
        "sudoku_progress_reward": progress_reward,
    }


# ---------------------------------------------------------------------------
# Env-gated reward functions (return None for N/A samples)
# ---------------------------------------------------------------------------


def reward_wordle_correct(completions: list[str], **kwargs) -> list[float | None]:
    """Wordle correctness reward. Returns None for non-Wordle samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("wordle_correct_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "wordle" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


def reward_wordle_greens(completions: list[str], **kwargs) -> list[float | None]:
    """Wordle green letters reward. Returns None for non-Wordle samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("wordle_green_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "wordle" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


def reward_wordle_yellows(completions: list[str], **kwargs) -> list[float | None]:
    """Wordle yellow letters reward. Returns None for non-Wordle samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("wordle_yellow_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "wordle" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


def reward_wordle_repetition(completions: list[str], **kwargs) -> list[float | None]:
    """Wordle repetition penalty. Returns None for non-Wordle samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("wordle_repetition_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "wordle" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


def reward_sudoku_correct(completions: list[str], **kwargs) -> list[float | None]:
    """Sudoku correctness reward. Returns None for non-Sudoku samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("sudoku_correct_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "sudoku" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


def reward_sudoku_valid_move(completions: list[str], **kwargs) -> list[float | None]:
    """Sudoku valid move reward. Returns None for non-Sudoku samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("sudoku_valid_move_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "sudoku" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


def reward_sudoku_empty_cell(completions: list[str], **kwargs) -> list[float | None]:
    """Sudoku empty cell targeting reward. Returns None for non-Sudoku samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("sudoku_empty_cell_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "sudoku" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


def reward_sudoku_repetition(completions: list[str], **kwargs) -> list[float | None]:
    """Sudoku repetition penalty. Returns None for non-Sudoku samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("sudoku_repetition_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "sudoku" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


def reward_sudoku_progress(completions: list[str], **kwargs) -> list[float | None]:
    """Sudoku progress reward. Returns None for non-Sudoku samples."""
    env_names = kwargs.get("env_name", [])
    rewards = kwargs.get("sudoku_progress_reward", [])
    result = []
    for i, _ in enumerate(completions):
        if i < len(env_names) and env_names[i] == "sudoku" and i < len(rewards):
            result.append(float(rewards[i]))
        else:
            result.append(None)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    tokenizer_id = args.tokenizer_id or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Create environment clients
    wordle_client = TextArenaEnv(base_url=args.wordle_env_url)
    sudoku_client = TextArenaEnv(base_url=args.sudoku_env_url)
    print(f"🌍 Wordle environment: {args.wordle_env_url}")
    print(f"🌍 Sudoku environment: {args.sudoku_env_url}")

    # Load system prompts
    wordle_system_prompt = resolve_system_prompt(args.wordle_system_prompt_path)
    sudoku_system_prompt = resolve_system_prompt(args.sudoku_system_prompt_path)

    # Build dataset with "env" column for routing
    wordle_entries = [{"prompt": args.wordle_dataset_prompt, "env": "wordle"} for _ in range(args.num_wordle)]
    sudoku_entries = [{"prompt": args.sudoku_dataset_prompt, "env": "sudoku"} for _ in range(args.num_sudoku)]
    all_entries = wordle_entries + sudoku_entries

    # Shuffle the dataset
    import random
    random.shuffle(all_entries)

    dataset = Dataset.from_list(all_entries)
    print(f"📊 Dataset: {args.num_wordle} Wordle + {args.num_sudoku} Sudoku = {len(dataset)} total samples")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"outputs/multi-env-grpo-{sanitize_name(args.model_id)}-{timestamp}")

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        max_completion_length=8,  # Small for single-turn moves
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        report_to="trackio",
        gradient_checkpointing=args.gradient_checkpointing,
    )

    grpo_config.run_name = args.run_name or f"run-{timestamp}"
    grpo_config.project = args.project or f"group-{sanitize_name(args.model_id)}"
    grpo_config.trackio_space_id = args.trackio_space_id

    def rollout_func(inputs: list[dict[str, Any]], trainer: GRPOTrainer) -> dict[str, list]:
        """Composite rollout function that routes samples to the appropriate environment."""
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_env_names: list[str] = []

        # Wordle rewards (None for non-Wordle samples)
        all_wordle_correct: list[float | None] = []
        all_wordle_green: list[float | None] = []
        all_wordle_yellow: list[float | None] = []
        all_wordle_repetition: list[float | None] = []

        # Sudoku rewards (None for non-Sudoku samples)
        all_sudoku_correct: list[float | None] = []
        all_sudoku_valid_move: list[float | None] = []
        all_sudoku_empty_cell: list[float | None] = []
        all_sudoku_repetition: list[float | None] = []
        all_sudoku_progress: list[float | None] = []

        for inp in inputs:
            env_name = inp["env"]
            all_env_names.append(env_name)

            if env_name == "wordle":
                episode = wordle_rollout_once(
                    trainer=trainer,
                    env=wordle_client,
                    tokenizer=tokenizer,
                    dataset_prompt=inp["prompt"],
                    system_prompt=wordle_system_prompt,
                    max_turns=args.wordle_max_turns,
                    api_delay=args.api_delay,
                )
                all_prompt_ids.append(episode["prompt_ids"])
                all_completion_ids.append(episode["completion_ids"])
                all_logprobs.append(episode["logprobs"])

                # Wordle rewards
                all_wordle_correct.append(episode["wordle_correct_reward"])
                all_wordle_green.append(episode["wordle_green_reward"])
                all_wordle_yellow.append(episode["wordle_yellow_reward"])
                all_wordle_repetition.append(episode["wordle_repetition_reward"])

                # Sudoku rewards are None for Wordle samples
                all_sudoku_correct.append(None)
                all_sudoku_valid_move.append(None)
                all_sudoku_empty_cell.append(None)
                all_sudoku_repetition.append(None)
                all_sudoku_progress.append(None)

            elif env_name == "sudoku":
                episode = sudoku_rollout_once(
                    trainer=trainer,
                    env=sudoku_client,
                    tokenizer=tokenizer,
                    system_prompt=sudoku_system_prompt,
                    max_turns=args.sudoku_max_turns,
                    difficulty=args.sudoku_difficulty,
                    api_delay=args.api_delay,
                    debug=args.debug,
                )
                all_prompt_ids.append(episode["prompt_ids"])
                all_completion_ids.append(episode["completion_ids"])
                all_logprobs.append(episode["logprobs"])

                # Wordle rewards are None for Sudoku samples
                all_wordle_correct.append(None)
                all_wordle_green.append(None)
                all_wordle_yellow.append(None)
                all_wordle_repetition.append(None)

                # Sudoku rewards
                all_sudoku_correct.append(episode["sudoku_correct_reward"])
                all_sudoku_valid_move.append(episode["sudoku_valid_move_reward"])
                all_sudoku_empty_cell.append(episode["sudoku_empty_cell_reward"])
                all_sudoku_repetition.append(episode["sudoku_repetition_reward"])
                all_sudoku_progress.append(episode["sudoku_progress_reward"])

            else:
                raise ValueError(f"Unknown environment: {env_name}")

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_name": all_env_names,
            # Wordle rewards
            "wordle_correct_reward": all_wordle_correct,
            "wordle_green_reward": all_wordle_green,
            "wordle_yellow_reward": all_wordle_yellow,
            "wordle_repetition_reward": all_wordle_repetition,
            # Sudoku rewards
            "sudoku_correct_reward": all_sudoku_correct,
            "sudoku_valid_move_reward": all_sudoku_valid_move,
            "sudoku_empty_cell_reward": all_sudoku_empty_cell,
            "sudoku_repetition_reward": all_sudoku_repetition,
            "sudoku_progress_reward": all_sudoku_progress,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            # Wordle rewards
            reward_wordle_correct,
            reward_wordle_greens,
            reward_wordle_yellows,
            reward_wordle_repetition,
            # Sudoku rewards
            reward_sudoku_correct,
            reward_sudoku_valid_move,
            reward_sudoku_empty_cell,
            reward_sudoku_repetition,
            reward_sudoku_progress,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print(f"🚀 Starting multi-env GRPO training:")
    print(f"   - {args.num_generations} generations per prompt")
    print(f"   - Wordle: {args.wordle_max_turns} max turns")
    print(f"   - Sudoku: {args.sudoku_max_turns} max turns, difficulty={args.sudoku_difficulty}")

    try:
        trainer.train()
    finally:
        wordle_client.close()
        sudoku_client.close()


if __name__ == "__main__":
    main()
