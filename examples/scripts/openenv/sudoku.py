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
#     "openenv-textarena @ git+https://huggingface.co/spaces/openenv/sudoku",
# ]
# ///

"""
GRPO training for Sudoku with TextArena environment.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/sudoku
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/textarena_env
uv pip install -e .
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/sudoku.py --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/sudoku.py --vllm-mode server --vllm-server-url http://localhost:8000
```

# Option 3: Local + Colocated vLLM (1 GPU required)

# Start the environment only if using --env-mode docker-local
```sh
docker run -d -p 8001:8001 registry.hf.space/openenv-sudoku:latest
```

```sh
python examples/scripts/openenv/sudoku.py --env-mode docker-local --vllm-mode colocate
```

# Full example with all flags:
```sh
python examples/scripts/openenv/sudoku.py \
    --vllm-mode colocate \
    --env-mode space \
    --env-host https://openenv-sudoku.hf.space \
    --num-generations 8 \
    --per-device-batch-size 1 \
    --max-turns 100 \
    --gradient-accumulation-steps 8 \
    --difficulty easy \
    --dataset-size 100
```
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions


# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from textarena_env import TextArenaAction, TextArenaEnv


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Sudoku")

    # Model
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")

    # Environment
    parser.add_argument("--env-host", type=str, default="https://openenv-sudoku.hf.space")
    parser.add_argument("--env-port", type=int, default=8001)
    parser.add_argument("--env-mode", choices=["docker-local", "docker-image", "docker-hub", "space"], default="space")
    parser.add_argument("--env-image", type=str, default="textarena-env:latest")

    # Prompts
    parser.add_argument("--system-prompt-path", default="sudoku_prompt.txt")
    parser.add_argument("--dataset-prompt", default="Play Sudoku like an expert.")
    parser.add_argument("--dataset-size", type=int, default=1000)

    # Game settings
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="hard",
        help="Training difficulty: easy=guaranteed+options, medium=only options, hard=no hints",
    )
    parser.add_argument(
        "--api-delay", type=float, default=0.0, help="Delay in seconds between API calls to avoid rate limiting"
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


def count_filled_cells(board_str: str) -> int:
    """Count the number of filled cells in the board."""
    if not is_valid_board_state(board_str):
        return 0
    grid = parse_board(board_str)
    return sum(1 for row in grid for cell in row if cell != 0)


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


def extract_empty_cells_with_candidates(
    board_str: str, sort_by_difficulty: bool = True
) -> list[tuple[int, int, set[int]]]:
    """Extract empty cells with their valid candidate numbers.

    Args:
        sort_by_difficulty: If True, sort by number of candidates (easiest first).
                           If False, keep natural order (top-left to bottom-right).
    """
    grid = parse_board(board_str)
    cells_with_candidates = []

    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                candidates = get_valid_numbers(grid, row, col)
                cells_with_candidates.append((row + 1, col + 1, candidates))  # 1-indexed

    if sort_by_difficulty:
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


def make_compact_prompt(
    board: str,
    step: int,
    successful_moves: list[str],
    failed_moves: list[str],
    difficulty: str = "hard",
) -> str:
    """Create a compact prompt with only essential info (saves tokens!).

    Args:
        difficulty: Training difficulty level:
            - "easy": Show guaranteed moves (naked singles) + other options
            - "medium": Only show other options (hints where to look, not exact answers)
            - "hard": No hints (model must learn Sudoku rules by itself)
    """

    # Summary line
    cells_filled = len(successful_moves)
    summary = f"Step {step}. Progress: {cells_filled} cells filled."

    # Board (only show the grid, stripped down)
    board_only = extract_board_only(board) if board else "No board available."

    # Moves already tried (for learning what NOT to do)
    tried_moves_hint = ""
    all_tried = successful_moves + failed_moves
    if all_tried:
        tried_moves_hint = f"\n\n⚠️ MOVES ALREADY TRIED (do not repeat): {', '.join(all_tried)}"

    # Hints based on difficulty
    hints = ""
    if difficulty == "easy" and board:
        # Easy: sorted by difficulty, show guaranteed moves + other easy options
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
        # Medium: NOT sorted, just show empty cells with candidates (no ordering hints)
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
    difficulty: str = "hard",
    api_delay: float = 0.0,
) -> dict[str, list]:
    result = env.reset()
    time.sleep(api_delay)  # Avoid rate limiting
    observation = result.observation

    # Only store the LAST turn for backprop (much more efficient!)
    last_turn_data: dict | None = None

    valid_move_scores: list[float] = []
    empty_cell_scores: list[float] = []
    correct_scores: list[float] = []
    repetition_scores: list[float] = []

    move_counts: defaultdict[str, int] = defaultdict(int)

    # Track successful and failed moves for summary
    successful_moves: list[str] = []
    failed_moves: list[str] = []

    # Extract initial board state
    last_board_state = ""
    initial_filled = 0
    for message in observation.messages:
        if message.content and is_valid_board_state(message.content):
            last_board_state = message.content
            initial_filled = count_filled_cells(last_board_state)
            break

    max_filled = initial_filled  # Track max progress

    for turn in range(max_turns):
        if result.done:
            break

        # Build COMPACT prompt (saves tokens!)
        user_prompt = make_compact_prompt(
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
            print(f"STEP {turn + 1}")
            print(f"{'=' * 60}")
            print(f"USER PROMPT:\n{user_prompt}")
            print(f"{'=' * 60}")

        # Generate
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]

        # Store ONLY this turn's data (replace previous)
        last_turn_data = {
            "prompt_ids": rollout_outputs["prompt_ids"],
            "completion_ids": rollout_outputs["completion_ids"],
            "logprobs": rollout_outputs["logprobs"],
        }

        if debug:
            step_tokens = len(rollout_outputs["prompt_ids"]) + len(rollout_outputs["completion_ids"])
            print(f"TOKENS: this_step={step_tokens} (only last turn used for backprop)")

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
        time.sleep(api_delay)  # Avoid rate limiting
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

        # Calculate valid_move_score and repetition_score
        is_new_move = move_counts[move] == 0
        repetition_count = move_counts[move]
        move_counts[move] += 1

        # Exponential penalty for repetitions: -2^(n-1) capped at -10
        # 1st repeat: -1, 2nd: -2, 3rd: -4, 4th+: -10 (capped)
        if repetition_count > 0:
            repetition_score = -min(2 ** (repetition_count - 1), 10.0)
        else:
            repetition_score = 0.0

        if debug:
            print(
                f"SCORES: empty_cell={empty_cell_score}, is_new={is_new_move}, repetitions={repetition_count}, rep_penalty={repetition_score}"
            )

        if not debug:
            print(f"Step {turn + 1}: {move}")

        if feedback["valid_move"] and is_new_move:
            valid_move_score = 1.0
            if move:
                successful_moves.append(move)  # Track for summary
        elif feedback["got_warning"]:
            valid_move_score = -0.5
            if move:
                failed_moves.append(move)  # Track for summary
        else:
            valid_move_score = 0.0

        # Update board state and track progress
        if feedback["board_state"] and is_valid_board_state(feedback["board_state"]):
            last_board_state = feedback["board_state"]
            current_filled = count_filled_cells(last_board_state)
            if current_filled > max_filled:
                max_filled = current_filled

        valid_move_scores.append(valid_move_score)
        empty_cell_scores.append(empty_cell_score)
        correct_scores.append(correct_score)
        repetition_scores.append(repetition_score)

    # Aggregate rewards
    correct_reward = correct_scores[-1] if correct_scores else 0.0
    valid_move_reward = sum(valid_move_scores) / len(valid_move_scores) if valid_move_scores else 0.0
    empty_cell_reward = sum(empty_cell_scores) / len(empty_cell_scores) if empty_cell_scores else 0.0
    repetition_reward = sum(repetition_scores) / len(repetition_scores) if repetition_scores else 0.0

    # Progress reward: how many cells we filled beyond initial state (normalized to 0-1)
    # 81 total cells, so (max_filled - initial_filled) / (81 - initial_filled) gives progress
    remaining_to_fill = 81 - initial_filled
    if remaining_to_fill > 0:
        progress_reward = (max_filled - initial_filled) / remaining_to_fill
    else:
        progress_reward = 1.0  # Already complete

    # Use ONLY last turn for backpropagation (much more efficient!)
    if last_turn_data:
        prompt_ids = last_turn_data["prompt_ids"]
        completion_ids = last_turn_data["completion_ids"]
        logprobs = last_turn_data["logprobs"]
    else:
        prompt_ids = []
        completion_ids = []
        logprobs = []

    total_tokens = len(prompt_ids) + len(completion_ids)
    cells_filled = max_filled - initial_filled
    print(
        f"Episode: empty_cell={empty_cell_reward:.2f}, valid={valid_move_reward:.2f}, "
        f"repetition={repetition_reward:.2f}, progress={progress_reward:.2f} ({cells_filled} cells), "
        f"correct={correct_reward:.2f}, tokens={total_tokens}"
    )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "correct_reward": correct_reward,
        "valid_move_reward": valid_move_reward,
        "empty_cell_reward": empty_cell_reward,
        "repetition_reward": repetition_reward,
        "progress_reward": progress_reward,
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


def reward_repetition(completions: list[str], **kwargs) -> list[float]:
    """Penalty for repeating moves."""
    rewards = kwargs.get("repetition_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_progress(completions: list[str], **kwargs) -> list[float]:
    """Reward for filling more cells in the board."""
    rewards = kwargs.get("progress_reward")
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
        report_to="trackio",
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
        all_repetition = []
        all_progress = []

        for _ in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=client,
                tokenizer=trainer.processing_class,
                system_prompt=system_prompt,
                max_turns=args.max_turns,
                debug=args.debug,
                difficulty=args.difficulty,
                api_delay=args.api_delay,
            )
            all_prompt_ids.append(episode["prompt_ids"])
            all_completion_ids.append(episode["completion_ids"])
            all_logprobs.append(episode["logprobs"])
            all_correct.append(episode["correct_reward"])
            all_valid.append(episode["valid_move_reward"])
            all_empty_cell.append(episode["empty_cell_reward"])
            all_repetition.append(episode["repetition_reward"])
            all_progress.append(episode["progress_reward"])

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "correct_reward": all_correct,
            "valid_move_reward": all_valid,
            "empty_cell_reward": all_empty_cell,
            "repetition_reward": all_repetition,
            "progress_reward": all_progress,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[
            reward_empty_cell,  # Learn to pick empty cells
            reward_valid_moves,  # Learn valid numbers
            reward_repetition,  # Penalize repeating moves
            reward_progress,  # Reward filling more cells
            reward_correct,  # Solve the puzzle
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
