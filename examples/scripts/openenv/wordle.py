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
#     "trl",
#     "trackio",
#     "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Wordle environment and vLLM.
"""

from datasets import Dataset
from textarena_env import TextArenaAction, TextArenaEnv

from trl import GRPOConfig, GRPOTrainer


prompt = """You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

Follow these rules to play Wordle:

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position
   - YELLOW (Y): Letter is in the word but in the wrong position
   - GRAY (X): Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed
6. Use the tool `guess` to make a guess.
"""


class WordleEnv:
    def __init__(self):
        self.client = TextArenaEnv(base_url="https://openenv-wordle.hf.space")

    def reset(self, **kwargs) -> None:
        self.client.reset()
        self.reward = -1.0
        # The game returns cumulative feedback each turn (new text appended at the end), so
        # we store the previous full response and slice out only the newly appended part.
        self._last_full_feedback = ""
        self.done = False

    def guess(self, guess: str) -> str:
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            self.reward = -1.0  # Penalize guesses after game is done
            raise ValueError("Game over.")
        result = self.client.step(TextArenaAction(message=guess))
        _full_feedback = result.observation.messages[0].content
        # Just take the new feedback since the last guess, which is the part appended to the end of the full feedback
        feedback = _full_feedback[len(self._last_full_feedback) :]
        self._last_full_feedback = _full_feedback
        # For some reason, the environment doesn't penalize invalid moves and just returns the last reward.
        # We check the feedback for the invalid move message and penalize it if found.
        if "You attempted an invalid move" in feedback:
            self.reward = -1.0  # Penalize invalid moves
        else:
            self.reward = result.reward
        self.done = result.done
        return feedback


def reward(environments, **kwargs) -> list[float]:
    return [environment.reward for environment in environments]


def main() -> None:
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}] for _ in range(1000)]})

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-1.7B",
        reward_funcs=reward,
        train_dataset=dataset,
        args=GRPOConfig(
            report_to="trackio",
            trackio_space_id="wordle-grpo",
            log_completions=True,
            num_completions_to_print=2,
            logging_steps=1,
            chat_template_kwargs={"enable_thinking": False},
            max_completion_length=1024,
        ),
        environment_factory=WordleEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
