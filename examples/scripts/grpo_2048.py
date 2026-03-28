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
#     "trl[peft]",
# ]
# ///

import random

from datasets import Dataset
from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer


PROMPT = "Play 2048 on a 4x4 board. Use the tool `move` with one of: up, down, left, right. Maximize the score."


class Game2048Env:
    def reset(self, **kwargs) -> str:
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0.0
        self.done = False
        self._spawn()
        self._spawn()
        return f"score={self.score}\n{self._render()}\ndone={self.done}"

    def move(self, direction: str) -> str:
        """
        Play one move in 2048.

        Args:
            direction: One of "up", "down", "left", "right".

        Returns:
            Environment feedback after the move.
        """
        if self.done:
            raise ValueError("Game over.")
        moved, gained = self._apply_move(direction.strip().lower())
        if moved:
            self.score += gained
            self._spawn()
        self.done = not self._can_move()
        return f"score={self.score}\n{self._render()}\ndone={self.done}"

    def _spawn(self) -> None:
        empty = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if not empty:
            return
        r, c = random.choice(empty)
        self.board[r][c] = 4 if random.random() < 0.1 else 2

    @staticmethod
    def _merge_line(line: list[int]) -> tuple[list[int], int]:
        vals = [x for x in line if x]
        out = []
        gained = 0
        i = 0
        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                v = vals[i] * 2
                out.append(v)
                gained += v
                i += 2
            else:
                out.append(vals[i])
                i += 1
        out += [0] * (4 - len(out))
        return out, gained

    def _apply_move(self, direction: str) -> tuple[bool, int]:
        if direction not in {"up", "down", "left", "right"}:
            return False, 0

        before = [row[:] for row in self.board]
        gained_total = 0

        if direction in {"left", "right"}:
            for r in range(4):
                row = self.board[r][:]
                if direction == "right":
                    row.reverse()
                merged, gained = self._merge_line(row)
                if direction == "right":
                    merged.reverse()
                self.board[r] = merged
                gained_total += gained
        else:
            for c in range(4):
                col = [self.board[r][c] for r in range(4)]
                if direction == "down":
                    col.reverse()
                merged, gained = self._merge_line(col)
                if direction == "down":
                    merged.reverse()
                for r in range(4):
                    self.board[r][c] = merged[r]
                gained_total += gained

        moved = self.board != before
        return moved, gained_total

    def _can_move(self) -> bool:
        if any(0 in row for row in self.board):
            return True
        for r in range(4):
            for c in range(4):
                if r + 1 < 4 and self.board[r][c] == self.board[r + 1][c]:
                    return True
                if c + 1 < 4 and self.board[r][c] == self.board[r][c + 1]:
                    return True
        return False

    def _render(self) -> str:
        return "\n".join(" ".join(f"{v:3d}" for v in row) for row in self.board)


def reward_score(environments, **kwargs):
    return [env.score for env in environments]


def main() -> None:
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": PROMPT}] for _ in range(1000)]})

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B",
        train_dataset=dataset,
        reward_funcs=reward_score,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=2,
            report_to="trackio",
            trackio_space_id="trl-2048",
            max_completion_length=2048,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
        ),
        environment_factory=Game2048Env,
        peft_config=LoraConfig(),
    )
    trainer.train()


if __name__ == "__main__":
    main()
