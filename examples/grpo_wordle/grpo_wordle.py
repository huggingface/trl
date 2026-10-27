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
#     "trackio",
#     "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env",
# ]
# ///


"""
GRPO training with OpenEnv's Wordle environment and vLLM.

The model starts from an SFT warmup of Qwen3-1.7B on solver-played games
([`openenv/Qwen3-1.7B-Wordle-SFT`](https://huggingface.co/openenv/Qwen3-1.7B-Wordle-SFT)): the base model
almost never wins, which leaves GRPO little to learn from. The OpenEnv Wordle tutorial covers the whole pipeline (data,
SFT, GRPO and evaluation): https://huggingface.co/docs/openenv/tutorials/wordle-grpo

Each step plays 64 games, one environment session each. Duplicate the `openenv/wordle` Space, set the
`MAX_CONCURRENT_ENVS=64` variable on your copy and pass its URL with `--env-url`.

# Option 1: Colocated vLLM (1 GPU required)
```sh
python examples/grpo_wordle/grpo_wordle.py --vllm-mode colocate --env-url https://<user>-wordle.hf.space
```

# Option 2: Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve openenv/Qwen3-1.7B-Wordle-SFT --host 0.0.0.0 --port 8000 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/grpo_wordle/grpo_wordle.py --vllm-mode server --vllm-server-url http://localhost:8000 \
    --env-url https://<user>-wordle.hf.space
```

# Option 3: Hugging Face Jobs (1x A100, about 2.5 hours)
```sh
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN examples/grpo_wordle/grpo_wordle.py \
    --env-url https://<user>-wordle.hf.space --push-to-hub
```
"""

import argparse
import re

from datasets import Dataset
from huggingface_hub import get_token
from textarena_env import TextArenaAction, TextArenaEnv

from trl import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Wordle using the TextArena OpenEnv environment."
    )
    parser.add_argument("--model", type=str, default="openenv/Qwen3-1.7B-Wordle-SFT", help="Model to train.")
    parser.add_argument("--env-url", type=str, default="https://openenv-wordle.hf.space", help="Environment URL.")
    parser.add_argument("--max-steps", type=int, default=150, help="Number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to store outputs and checkpoints.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the trained model to the Hub.")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate", help="vLLM execution mode.")
    parser.add_argument(
        "--vllm-server-url", type=str, default="http://localhost:8000", help="vLLM server URL (server mode only)."
    )
    return parser.parse_args()


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

FEEDBACK = re.compile(r"Feedback:\n([A-Z ]+)\n([GYX ]+)")


# The environment pays a win the same at any speed and pays partial credit only at the end of a game, so a policy can
# learn to think less and spend every guess. These terms reward winning fast, give partial credit from the best guess
# whether or not the game finished, and penalize invalid moves.
def reward_win(environments, **kwargs) -> list[float]:
    return [1.0 + 0.5 * (1 - env.num_guesses / 6) if env.won else 0.0 for env in environments]


def reward_partial(environments, **kwargs) -> list[float]:
    return [0.0 if env.won else 0.1 * env.best_greens for env in environments]


def reward_invalid(environments, **kwargs) -> list[float]:
    return [-0.1 * env.invalid_moves for env in environments]


def main() -> None:
    args = parse_args()

    env_url = args.env_url
    # A private Space needs your token. A public one ignores it.
    headers = {"Authorization": f"Bearer {get_token()}"} if get_token() else None

    class WordleEnv:
        def __init__(self):
            self.client = TextArenaEnv(base_url=env_url, headers=headers)

        def reset(self, **kwargs) -> str | None:
            result = self.client.reset()
            # The game returns cumulative feedback each turn (new text appended at the end), so
            # we store the previous full response and slice out only the newly appended part.
            self._last_full_feedback = result.observation.messages[0].content
            self.done = False
            self.won = False
            self.num_guesses = 0
            self.best_greens = 0
            self.invalid_moves = 0
            return self._last_full_feedback

        def guess(self, guess: str) -> str:
            """
            Make a guess in the Wordle environment.

            Args:
                guess: The guessed word, formatted as '[abcde]'

            Returns:
                The feedback message from the environment.
            """
            if self.done:
                raise ValueError("Game over.")
            result = self.client.step(TextArenaAction(message=guess))
            _full_feedback = result.observation.messages[0].content
            # Just take the new feedback since the last guess, which is the part appended to the end of the full feedback
            feedback = _full_feedback[len(self._last_full_feedback) :]
            self._last_full_feedback = _full_feedback
            if "invalid move" in feedback.lower():
                self.invalid_moves += 1
            else:
                self.num_guesses += 1
                match = FEEDBACK.search(feedback)
                if match:
                    self.best_greens = max(self.best_greens, match.group(2).count("G"))
                self.won = result.done and result.reward >= 1.0
            self.done = result.done
            return feedback

    output_dir = args.output_dir or f"{args.model.split('/')[-1]}-GRPO"
    # 4 prompts x 16 games per step
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * (args.max_steps * 4)})

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_win, reward_partial, reward_invalid],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=output_dir,
            use_vllm=True,
            vllm_mode=args.vllm_mode,
            vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
            vllm_gpu_memory_utilization=0.35,
            report_to="trackio",
            log_completions=True,
            num_completions_to_print=1,
            logging_steps=1,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type="constant",
            num_generations=16,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            bf16=True,
            chat_template_kwargs={"enable_thinking": True},
            max_completion_length=4096,
            mask_truncated_completions=True,
            # Multi-turn rollouts: clip per-token ratios instead of masking whole sequences.
            vllm_importance_sampling_mode="token_truncate",
            max_tool_calling_iterations=6,
            push_to_hub=args.push_to_hub,
        ),
        environment_factory=WordleEnv,
    )
    trainer.train()
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
