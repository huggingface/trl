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
#     "vllm>=0.33.0",
#     "chess",
# ]
# ///

# docstyle-ignore
"""
Share of the 1,000 test puzzles a model solves, and its puzzle rating: its move must be the first move of the
solution, or any move that mates, as Lichess accepts. One GPU.

    python examples/sft_gpt_oss_chess/evaluate.py gpt-oss-20b-chess-mxfp4

The original model reasons before it answers. At its default reasoning effort, most of its answers do not finish
within the 8k-token budget, so evaluate it with a low effort:

    python examples/sft_gpt_oss_chess/evaluate.py openai/gpt-oss-20b --reasoning_effort low
"""

import argparse
import re

import chess
from datasets import load_dataset
from vllm import LLM, SamplingParams


def solves(prompt, solution, answer):
    board = chess.Board(re.search(r"FEN: (.+)", prompt).group(1))
    # The first token that reads as a legal move is the answer, in UCI (e2e4) or SAN (Nf3) notation.
    for token in answer.replace("*", " ").split():
        token = token.strip(".,;:!?()`'\"").rstrip("+#")
        for parse in (board.parse_uci, board.parse_san):
            try:
                move = parse(token)
            except ValueError:
                continue
            if move.uci() == solution:
                return True
            board.push(move)
            return board.is_checkmate()
    return False


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"])
args = parser.parse_args()

dataset = load_dataset("trl-lib/chess-puzzles", split="test")
llm = LLM(args.model, dtype="bfloat16", max_model_len=10240, seed=0)
outputs = llm.chat(
    [example["prompt"] for example in dataset],
    SamplingParams(temperature=0.0, max_tokens=8192, skip_special_tokens=False),
    chat_template_kwargs={"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None,
)
solved = 0
for example, output in zip(dataset, outputs, strict=True):
    # gpt-oss writes its answer in the final channel, after its reasoning in the analysis channel.
    text = output.outputs[0].text
    if "<|channel|>final<|message|>" in text:
        answer = text.rsplit("<|channel|>final<|message|>", 1)[1].split("<|return|>")[0]
        solved += solves(example["prompt"][0]["content"], example["completion"][0]["content"], answer)

# Each puzzle has a Lichess rating, like an opponent. The model's puzzle rating is the one at which the Elo formula
# expects it to solve as many puzzles as it did (the maximum-likelihood rating).
low, high = 0.0, 4000.0
for _ in range(50):
    rating = (low + high) / 2
    expected = sum(1 / (1 + 10 ** ((puzzle_rating - rating) / 400)) for puzzle_rating in dataset["Rating"])
    low, high = (rating, high) if expected < solved else (low, rating)
print(f"{args.model}: {100 * solved / len(dataset):.1f}% of the puzzles solved, puzzle rating {rating:.0f}")
