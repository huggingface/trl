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
GRPO training for Sudoku with TextArena environment.

Usage:
    python examples/scripts/openenv/sudoku.py \
        --vllm-mode colocate \
        --env-mode space \
        --env-host https://sergiopaniego-textarena.hf.space \
        --num-generations 8 \
        --per-device-batch-size 1 \
        --max-turns 50 \
        --gradient-accumulation-steps 8  \
        --keep-last-n-turns 25xw
"""

from __future__ import annotations

import argparse
import re
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Sudoku")

    # Model
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")

    # Environment
    parser.add_argument("--env-host", type=str, default="0.0.0.0")
    parser.add_argument("--env-port", type=int, default=8001)
    parser.add_argument(
        "--env-mode", choices=["docker-local", "docker-image", "docker-hub", "space"], default="docker-image"
    )
    parser.add_argument("--env-image", type=str, default="textarena-env:latest")

    # Prompts
    parser.add_argument("--system-prompt-path", default="sudoku_prompt.txt")
    parser.add_argument("--dataset-prompt", default="Play Sudoku like an expert.")
    parser.add_argument("--dataset-size", type=int, default=3000)

    # Game settings
    parser.add_argument("--max-turns", type=int, default=31)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--keep-last-n-turns", type=int, default=3, help="Only keep last N turns for backpropagation (saves memory)"
    )

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=None)

    # Training
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=1)

    # Checkpoints
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)

    # Logging
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--trackio-space-id", default="Sudoku-GRPO")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory",
    )

    # vLLM
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-server-url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2)

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


def extract_sudoku_move(text: str) -> str:
    """Extract a Sudoku move [row col number] from text."""
    # Try with spaces
    match = re.search(r"\[(\d)\s+(\d)\s+(\d)\]", text)
    if match:
        row, col, num = match.groups()
        return f"[{row} {col} {num}]"

    # Try without spaces
    match = re.search(r"\[(\d)(\d)(\d)\]", text)
    if match:
        row, col, num = match.groups()
        return f"[{row} {col} {num}]"

    return ""


def is_valid_board_state(board_str: str) -> bool:
    """Check if the string contains an actual Sudoku board."""
    return "R1" in board_str and "R9" in board_str and "|" in board_str


def parse_board(board_str: str) -> list[list[int]]:
    """Parse board string into 9x9 grid (0 = empty)."""
    grid = [[0] * 9 for _ in range(9)]
    if not is_valid_board_state(board_str):
        return grid

    for line in board_str.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped[0] == "R" and len(line_stripped) > 1 and line_stripped[1].isdigit():
            row = int(line_stripped[1]) - 1  # 0-indexed
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


def get_valid_numbers(grid: list[list[int]], row: int, col: int) -> set[int]:
    """Get valid numbers for a cell based on Sudoku rules."""
    if grid[row][col] != 0:
        return set()

    used = set()

    # Check row
    for c in range(9):
        if grid[row][c] != 0:
            used.add(grid[row][c])

    # Check column
    for r in range(9):
        if grid[r][col] != 0:
            used.add(grid[r][col])

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] != 0:
                used.add(grid[r][c])

    return set(range(1, 10)) - used


def extract_empty_cells_with_candidates(board_str: str) -> list[tuple[int, int, set[int]]]:
    """Extract empty cells with their valid candidate numbers."""
    grid = parse_board(board_str)
    cells_with_candidates = []

    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                candidates = get_valid_numbers(grid, row, col)
                cells_with_candidates.append((row + 1, col + 1, candidates))  # 1-indexed

    # Sort by number of candidates (easiest first = naked singles)
    cells_with_candidates.sort(key=lambda x: len(x[2]))

    return cells_with_candidates


def extract_empty_cells(board_str: str) -> list[tuple[int, int]]:
    """Extract list of empty cells (row, col) from board string."""
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
    """Extract just the Sudoku grid from a message."""
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


def format_history(messages: Iterable[TextArenaMessage]) -> str:
    lines = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if content:
            lines.append(f"[{tag}] {content}")
    return "\n".join(lines)


def make_user_prompt(messages: Iterable[TextArenaMessage], last_board: str = "") -> str:
    """Create prompt with board, empty cells, and previous moves."""
    messages_list = list(messages)
    history = format_history(messages_list)
    history_section = history if history else "Game starting."

    # Extract previous moves made by the model
    previous_moves = []
    for message in messages_list:
        if message.sender_id == 0 and message.content:
            move = extract_sudoku_move(message.content)
            if move and move not in previous_moves:
                previous_moves.append(move)

    previous_moves_hint = ""
    if previous_moves:
        moves_str = ", ".join(previous_moves)
        previous_moves_hint = f"\n\nMoves you already tried (DO NOT REPEAT): {moves_str}"

    # Check if history already contains board
    history_has_board = is_valid_board_state(history_section)

    # Get board to use
    board_to_use = ""
    for message in reversed(messages_list):
        if message.content and is_valid_board_state(message.content):
            board_to_use = message.content
            break
    if not board_to_use and last_board:
        board_to_use = last_board

    # Build hints
    candidates_hint = ""
    board_hint = ""
    if board_to_use:
        cells_with_candidates = extract_empty_cells_with_candidates(board_to_use)
        if cells_with_candidates:
            # Show cells with fewest candidates first (easiest moves)
            hints = []
            for row, col, candidates in cells_with_candidates[:8]:  # Show top 8 easiest
                if len(candidates) == 1:
                    # Naked single - only one option!
                    num = list(candidates)[0]
                    hints.append(f"({row},{col})→{num} ONLY OPTION!")
                else:
                    nums = ",".join(str(n) for n in sorted(candidates))
                    hints.append(f"({row},{col})→{nums}")

            remaining = len(cells_with_candidates) - 8 if len(cells_with_candidates) > 8 else 0
            candidates_hint = "\n\nEasiest cells to fill (valid numbers shown):\n" + " | ".join(hints)
            if remaining > 0:
                candidates_hint += f"\n...and {remaining} more cells."

        if not history_has_board:
            board_only = extract_board_only(board_to_use)
            if board_only:
                board_hint = f"\n\nCurrent board:\n{board_only}"

    return f"{history_section}{previous_moves_hint}{candidates_hint}{board_hint}\n\nYour move:"


def check_move_targets_empty_cell(move: str, board_str: str) -> bool:
    """Check if the move targets an empty cell on the board."""
    if not move or not board_str:
        return False

    match = re.search(r"\[(\d)\s+(\d)\s+(\d)\]", move)
    if not match:
        return False

    row, col = int(match.group(1)), int(match.group(2))
    empty_cells = extract_empty_cells(board_str)
    return (row, col) in empty_cells


def extract_feedback(observation) -> dict:
    """Extract feedback from environment observation."""
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


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def rollout_once(
    trainer: GRPOTrainer,
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_turns: int,
    debug: bool = False,
    keep_last_n_turns: int = 3,  # Only keep last N turns for backprop to save memory
) -> dict[str, list]:
    result = env.reset()
    observation = result.observation

    # Store tokens per turn (sliding window)
    turn_data: list[dict] = []

    valid_move_scores: list[float] = []
    empty_cell_scores: list[float] = []
    correct_scores: list[float] = []

    move_counts: defaultdict[str, int] = defaultdict(int)

    # Extract initial board state
    last_board_state = ""
    for message in observation.messages:
        if message.content and is_valid_board_state(message.content):
            last_board_state = message.content
            break

    for turn in range(max_turns):
        if result.done:
            break

        # Build prompt
        user_prompt = make_user_prompt(observation.messages, last_board_state)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        if debug:
            print(f"\n{'=' * 60}")
            print(f"STEP {turn + 1}")
            print(f"{'=' * 60}")
            print(f"USER PROMPT:\n{user_prompt}")
            print(f"{'=' * 60}")

        # Generate
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]

        # Store this turn's data
        turn_data.append(
            {
                "prompt_ids": rollout_outputs["prompt_ids"],
                "completion_ids": rollout_outputs["completion_ids"],
                "logprobs": rollout_outputs["logprobs"],
            }
        )

        # Keep only last N turns (sliding window to save memory)
        if len(turn_data) > keep_last_n_turns:
            turn_data.pop(0)

        # Calculate current totals for display
        total_prompt = sum(len(t["prompt_ids"]) for t in turn_data)
        total_completion = sum(len(t["completion_ids"]) for t in turn_data)

        if debug:
            step_tokens = len(rollout_outputs["prompt_ids"]) + len(rollout_outputs["completion_ids"])
            print(
                f"TOKENS: this_step={step_tokens}, window={len(turn_data)} turns, total_in_window={total_prompt}+{total_completion}"
            )

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Extract move
        move = extract_sudoku_move(completion_text)

        if debug:
            print(f"MODEL OUTPUT: {completion_text}")
            print(f"EXTRACTED MOVE: {move}")

        # Step environment
        result = env.step(TextArenaAction(message=move))
        observation = result.observation
        correct_score = float(result.reward or 0.0)

        # Get feedback
        feedback = extract_feedback(observation)

        # Get environment response
        env_response = ""
        for msg in observation.messages:
            if msg.sender_id == -1:  # Environment message
                env_response = msg.content
                break

        if debug:
            print(
                f"ENV RESPONSE: {env_response[:200]}..."
                if len(env_response) > 200
                else f"ENV RESPONSE: {env_response}"
            )
            print(f"VALID: {feedback['valid_move']}, WARNING: {feedback['got_warning']}, REWARD: {correct_score}")

        # Calculate empty_cell_score
        if last_board_state and move:
            targets_empty = check_move_targets_empty_cell(move, last_board_state)
            empty_cell_score = 1.0 if targets_empty else -1.0
        else:
            empty_cell_score = 0.0

        # Calculate valid_move_score
        is_new_move = move_counts[move] == 0
        move_counts[move] += 1

        if debug:
            print(f"SCORES: empty_cell={empty_cell_score}, is_new={is_new_move}")

        if not debug:
            print(f"Step {turn + 1}: {move}")

        if feedback["valid_move"] and is_new_move:
            valid_move_score = 1.0
        elif feedback["got_warning"]:
            valid_move_score = -0.5
        else:
            valid_move_score = 0.0

        # Update board state
        if feedback["board_state"] and is_valid_board_state(feedback["board_state"]):
            last_board_state = feedback["board_state"]

        valid_move_scores.append(valid_move_score)
        empty_cell_scores.append(empty_cell_score)
        correct_scores.append(correct_score)

    # Aggregate rewards
    correct_reward = correct_scores[-1] if correct_scores else 0.0
    valid_move_reward = sum(valid_move_scores) / len(valid_move_scores) if valid_move_scores else 0.0
    empty_cell_reward = sum(empty_cell_scores) / len(empty_cell_scores) if empty_cell_scores else 0.0

    # Flatten turn_data (only last N turns) for backpropagation
    prompt_ids = []
    completion_ids = []
    logprobs = []
    for t in turn_data:
        prompt_ids.extend(t["prompt_ids"])
        completion_ids.extend(t["completion_ids"])
        logprobs.extend(t["logprobs"])

    total_tokens = len(prompt_ids) + len(completion_ids)
    print(
        f"Episode: empty_cell={empty_cell_reward:.2f}, valid={valid_move_reward:.2f}, correct={correct_reward:.2f}, tokens={total_tokens}"
    )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "correct_reward": correct_reward,
        "valid_move_reward": valid_move_reward,
        "empty_cell_reward": empty_cell_reward,
    }


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def reward_empty_cell(completions: list[str], **kwargs) -> list[float]:
    """Reward for targeting empty cells (learn to pick valid positions first)."""
    rewards = kwargs.get("empty_cell_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_valid_moves(completions: list[str], **kwargs) -> list[float]:
    """Reward for making valid moves."""
    rewards = kwargs.get("valid_move_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_correct(completions: list[str], **kwargs) -> list[float]:
    """Reward for solving the puzzle."""
    rewards = kwargs.get("correct_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Setup environment
    if args.env_mode == "docker-local":
        client = TextArenaEnv(base_url=f"http://{args.env_host}:{args.env_port}")
    elif args.env_mode == "docker-image":
        client = TextArenaEnv.from_docker_image(args.env_image)
    elif args.env_mode == "docker-hub":
        client = TextArenaEnv.from_hub(args.env_image)
    elif args.env_mode == "space":
        client = TextArenaEnv(base_url=args.env_host)
    else:
        raise ValueError(f"Unknown environment mode: {args.env_mode}")

    print(f"🌍 Environment: {args.env_mode}")

    system_prompt = resolve_system_prompt(args.system_prompt_path)
    dataset = Dataset.from_dict({"prompt": [args.dataset_prompt] * args.dataset_size})

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"outputs/sudoku-grpo-{sanitize_name(args.model_id)}-{timestamp}")

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization
        if args.vllm_gpu_memory_utilization
        else 0.2,  # Lower to leave more VRAM for backpropagation
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
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        # chat_template_kwargs={"enable_thinking": False},
    )

    grpo_config.run_name = args.run_name or f"run-{timestamp}"
    grpo_config.project = args.project or f"group-{sanitize_name(args.model_id)}"
    grpo_config.trackio_space_id = args.trackio_space_id
    grpo_config.gradient_checkpointing = args.gradient_checkpointing

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_correct = []
        all_valid = []
        all_empty_cell = []

        for _ in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=client,
                tokenizer=trainer.processing_class,
                system_prompt=system_prompt,
                max_turns=args.max_turns,
                debug=args.debug,
                keep_last_n_turns=args.keep_last_n_turns,
            )
            all_prompt_ids.append(episode["prompt_ids"])
            all_completion_ids.append(episode["completion_ids"])
            all_logprobs.append(episode["logprobs"])
            all_correct.append(episode["correct_reward"])
            all_valid.append(episode["valid_move_reward"])
            all_empty_cell.append(episode["empty_cell_reward"])

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "correct_reward": all_correct,
            "valid_move_reward": all_valid,
            "empty_cell_reward": all_empty_cell,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[
            reward_empty_cell,  # First: learn to pick empty cells
            reward_valid_moves,  # Then: learn valid numbers
            reward_correct,  # Finally: solve the puzzle
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print(f"🚀 Starting GRPO training: {args.num_generations} generations, {args.max_turns} max turns")

    try:
        trainer.train()
    finally:
        client.close()


if __name__ == "__main__":
    main()
