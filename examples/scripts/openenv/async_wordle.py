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
Async GRPO training with Wordle environment using delta weight sync to a remote vLLM HF Space.

Architecture:
    Local (1 GPU):   AsyncGRPOTrainer + rollout worker (env tool calls run locally)
    Remote Space 1:  vLLM server with DeltaWorkerExtension (GPU, serves /v1/completions)
    Remote Space 2:  TextArena Wordle game server (no GPU, already at openenv-wordle.hf.space)
    HF Hub Bucket:   Stores weight anchors and sparse deltas

# Option 1: Remote vLLM Space + Remote Wordle Space (fully remote inference)

## Step 1: Deploy vLLM on HF Spaces

The Dockerfile and README.md are provided in `examples/scripts/openenv/vllm_space/`.
Deploy with the HF CLI:

```sh
# Create the Space (l4x1 GPU, Docker SDK)
hf repos create <your-username>/vllm-wordle-inference \\
    --type space --space-sdk docker --flavor l4x1 \\
    --secrets HF_TOKEN=$HF_TOKEN \\
    --env VLLM_SERVER_DEV_MODE=1

# Upload Dockerfile + README
hf upload <your-username>/vllm-wordle-inference \\
    examples/scripts/openenv/vllm_space/ --type space

# Check status
hf spaces info <your-username>/vllm-wordle-inference
```

## Step 2: Run training locally (1 GPU)

```sh
CUDA_VISIBLE_DEVICES=0 python examples/scripts/openenv/async_wordle.py \\
    --vllm-server-url https://<your-vllm-space>.hf.space \\
    --env-url https://openenv-wordle.hf.space \\
    --delta-sync-repo-id <your-hf-username>/wordle-deltas \\
    --model Qwen/Qwen3-1.7B
```

# Option 2: Local vLLM + Remote Wordle Space (for testing)

## Terminal 1: Spin up local vLLM server

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-1.7B \\
    --worker-extension-cls trl.experimental.async_grpo.delta_engine.DeltaWorkerExtension \\
    --weight-transfer-config '{"backend":"nccl"}' \\
    --max-model-len 8192 \\
    --enforce-eager \\
    --gpu-memory-utilization 0.8 \\
    --logprobs-mode processed_logprobs
```

## Terminal 2: Run training

```sh
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/async_wordle.py \\
    --vllm-server-url http://localhost:8000 \\
    --delta-sync-repo-id <your-hf-username>/wordle-deltas \\
    --model Qwen/Qwen3-1.7B
```
"""

import argparse
import logging
import os

from datasets import Dataset
from textarena_env import TextArenaAction, TextArenaEnv

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async GRPO training for Wordle with delta weight sync to a remote vLLM HF Space."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model identifier passed to AsyncGRPOTrainer for fine-tuning.",
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default="https://openenv-wordle.hf.space",
        help="URL for the Wordle environment server.",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000",
        help="URL for the vLLM server (local or remote HF Space).",
    )
    parser.add_argument(
        "--delta-sync-repo-id",
        type=str,
        default=None,
        help="HF Hub bucket for delta weight patches (e.g. 'user/wordle-deltas'). Required.",
    )
    parser.add_argument(
        "--delta-sync-anchor-interval",
        type=int,
        default=10,
        help="Upload a full anchor checkpoint every N weight syncs.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of entries in the synthetic training dataset.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Number of rollout generations per prompt.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=32,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of training steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--max-staleness",
        type=int,
        default=5,
        help="Drop rollout samples generated more than this many weight versions ago.",
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
        default="aminediroHF/async_wordle_trackio",
        help="Trackio space identifier for logging.",
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
            self.client = TextArenaEnv(base_url=env_url).sync()
            self.reward = 0.0
            self.done = False

        def _reconnect(self):
            self.client = TextArenaEnv(base_url=env_url).sync()

        def reset(self, **kwargs) -> str | None:
            try:
                result = self.client.reset()
            except Exception:
                self._reconnect()
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
            try:
                result = self.client.step(TextArenaAction(message=guess))
            except Exception:
                self._reconnect()
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

    output_dir = args.output_dir or f"{args.model.split('/')[-1]}-async-wordle-GRPO"
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}] for _ in range(args.dataset_size)]})

    config = AsyncGRPOConfig(
        delta_sync_enabled=True,
        delta_sync_repo_id=args.delta_sync_repo_id,
        delta_sync_anchor_interval=args.delta_sync_anchor_interval,
        vllm_server_base_url=args.vllm_server_url,
        learning_rate=args.learning_rate,
        bf16=True,
        output_dir=output_dir,
        max_completion_length=1024,
        max_tool_calling_iterations=3,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_staleness=args.max_staleness,
        max_steps=args.max_steps,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        report_to="trackio",
        trackio_space_id=args.trackio_space_id,
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=WordleEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
