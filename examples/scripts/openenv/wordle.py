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

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/wordle
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/wordle.py --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle.py --vllm-mode server --vllm-server-url http://localhost:8000
```

# Option 3: Local Environment + Colocated vLLM (1 GPU required)

To run the Wordle environment locally, you have several options:

## Option 3a: Using Docker Image (Recommended)

First, build the Docker image from the textarena_env directory:
```sh
cd 3rd_party/OpenEnv/envs/textarena_env
docker build -t textarena-env:latest -f server/Dockerfile .
```

Then run the environment server:
```sh
docker run -d -p 8001:8001 textarena-env:latest
```

Finally, run training pointing to local server:
```sh
python examples/scripts/openenv/wordle.py --vllm-mode colocate --env-url http://localhost:8001
```

## Option 3b: Running Server Directly

From the textarena_env directory:
```sh
cd 3rd_party/OpenEnv/envs/textarena_env
uv venv && source .venv/bin/activate
uv pip install -e .
python -m uvicorn server.app:app --reload --port 8001
```

Then in another terminal, run training:
```sh
python examples/scripts/openenv/wordle.py --vllm-mode colocate --env-url http://localhost:8001
```

## Option 3c: Using Pre-built HF Space Image

```sh
docker run -d -p 8001:8001 registry.hf.space/burtenshaw-wordle:latest
python examples/scripts/openenv/wordle.py --vllm-mode colocate --env-url http://localhost:8001
```
"""

import argparse

from datasets import Dataset
from textarena_env import TextArenaAction, TextArenaEnv

from trl import GRPOConfig, GRPOTrainer, RichProgressCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Wordle using the TextArena OpenEnv environment."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model identifier passed to GRPOTrainer for fine-tuning.",
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default="https://openenv-wordle.hf.space",
        help="URL for the environment server.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of entries to include in the synthetic training dataset.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of rollout generations per dataset prompt.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for GRPO training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=64,
        help="Gradient accumulation steps for GRPO training.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Frequency of logging steps for GRPO training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where training outputs and checkpoints are stored.",
    )
    parser.add_argument(
        "--trackio-space-id",
        type=str,
        default="wordle-grpo",
        help="Trackio space identifier.",
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


def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]


def main() -> None:
    args = parse_args()

    env_url = args.env_url

    class WordleEnv:
        def __init__(self):
            self.client = TextArenaEnv(base_url=env_url)

        def reset(self, **kwargs) -> str | None:
            result = self.client.reset()
            # The game returns cumulative feedback each turn (new text appended at the end), so
            # we store the previous full response and slice out only the newly appended part.
            self._last_full_feedback = result.observation.messages[0].content
            self.reward = 0.0
            self.done = False
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
            # For some reason, the environment doesn't penalize invalid moves and just returns the last reward.
            # We check the feedback for the invalid move message and penalize it if found.
            if "You attempted an invalid move" in feedback:
                self.reward = 0.0
            else:
                self.reward = result.reward
            self.done = result.done
            return feedback

    output_dir = args.output_dir or f"{args.model.split('/')[-1]}-wordle-GRPO"
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}] for _ in range(args.dataset_size)]})

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=output_dir,
            use_vllm=True,
            vllm_mode=args.vllm_mode,
            vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
            report_to="trackio",
            trackio_space_id=args.trackio_space_id,
            log_completions=True,
            num_completions_to_print=2,
            logging_steps=args.logging_steps,
            num_train_epochs=args.num_epochs,
            num_generations=args.num_generations,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            chat_template_kwargs={"enable_thinking": False},
            max_completion_length=1024,
        ),
        environment_factory=WordleEnv,
        callbacks=[RichProgressCallback()],
    )
    trainer.train()


if __name__ == "__main__":
    main()
