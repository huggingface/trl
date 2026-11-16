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

from dataclasses import dataclass, field

import chess
from datasets import load_dataset
from huggingface_hub import ModelCard
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-lib/chess-puzzles"`):
            Hugging Face repository ID to push the dataset to.
        dataset_num_proc (`int`, *optional*):
            Number of workers to use for dataset processing.
    """

    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-lib/chess-puzzles",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of workers to use for dataset processing."},
    )


def to_prompt_completion(example):
    board = chess.Board(example["FEN"])
    moves = example["Moves"].split()
    board.push_uci(moves[0])  # the opponent's move that sets up the puzzle
    side = "White" if board.turn == chess.WHITE else "Black"
    legal_moves = " ".join(sorted(move.uci() for move in board.legal_moves))
    prompt = (
        f"Chess puzzle. {side} to move.\n\n{board}\n\nFEN: {board.fen()}\nLegal moves (UCI): {legal_moves}\n\n"
        "Find the best move. Answer with the move in UCI notation only."
    )
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": moves[1]}],
    }


model_card = ModelCard("""
---
tags: [trl]
license: cc0-1.0
---

# Chess Puzzles Dataset

## Summary

The Chess Puzzles dataset is a processed version of the [Lichess puzzle database](https://huggingface.co/datasets/Lichess/chess-puzzles): 6 million positions from real games, each with a single winning move, rated by how hard Lichess players found them. Each puzzle is a prompt showing the board, its FEN and the legal moves, and a completion with the first move of the solution in UCI notation.

## Data Structure

- **Format**: [Conversational](https://huggingface.co/docs/trl/main/dataset_formats#conversational)
- **Type**: [Prompt-completion](https://huggingface.co/docs/trl/main/dataset_formats#prompt-completion)

Columns:
- `"prompt"`: The puzzle: the side to move, the board, its FEN and the legal moves.
- `"completion"`: The first move of the solution, in UCI notation (for example `e2e4`).
- `"PuzzleId"`, `"Rating"`, `"Themes"`: The Lichess puzzle ID, its rating, and its themes (`mateIn2`, `fork`, ...).

The test split holds 1,000 puzzles. Lichess also accepts any move that mates, so a model that answers with another mating move solves the puzzle too.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/chess_puzzles.py).
""")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("Lichess/chess-puzzles", split="train")
    dataset = dataset.map(
        to_prompt_completion,
        num_proc=script_args.dataset_num_proc,
        remove_columns=["GameId", "FEN", "Moves", "RatingDeviation", "Popularity", "NbPlays", "OpeningTags"],
    )
    dataset = dataset.train_test_split(test_size=1_000, seed=0)

    if script_args.push_to_hub:
        dataset.push_to_hub(script_args.repo_id)
        model_card.push_to_hub(script_args.repo_id, repo_type="dataset")
