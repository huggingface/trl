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
GRPO training with OpenEnv's Sudoku environment and vLLM.

The model starts from an SFT warmup of Qwen3-1.7B on solver-played games
([`openenv/Qwen3-1.7B-Sudoku-SFT`](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT)): the base model solves no
puzzle, which leaves GRPO little to learn from. The OpenEnv Sudoku tutorial covers the whole pipeline (data, SFT, GRPO
and evaluation): https://huggingface.co/docs/openenv/tutorials/sudoku-grpo

Each step plays 64 games, one environment session each. Duplicate the `openenv/sudoku` Space and set these variables on
your copy: `TEXTARENA_KW_CLUES=55` (about 26 empty cells, the Space default is 60 clues, about 21) and
`MAX_CONCURRENT_ENVS=96` (64 games, plus room for the sessions being closed between games). Pass its URL with
`--env-url`.

# Option 1: Colocated vLLM (1 GPU required)
```sh
python examples/grpo_sudoku/grpo_sudoku.py --vllm-mode colocate --env-url https://<user>-sudoku.hf.space
```

# Option 2: Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve openenv/Qwen3-1.7B-Sudoku-SFT --host 0.0.0.0 --port 8000 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1
```

# Run training (Terminal 2)
```sh
CUDA_VISIBLE_DEVICES=1 python examples/grpo_sudoku/grpo_sudoku.py --vllm-mode server --vllm-server-url http://localhost:8000 \
    --env-url https://<user>-sudoku.hf.space
```

# Option 3: Hugging Face Jobs (1x A100, about 15 hours)
```sh
hf jobs uv run --flavor a100-large --timeout 18h --secrets HF_TOKEN examples/grpo_sudoku/grpo_sudoku.py \
    --env-url https://<user>-sudoku.hf.space --push-to-hub
```
"""

import argparse

from datasets import Dataset
from huggingface_hub import get_token
from textarena_env import TextArenaAction, TextArenaEnv

from trl import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Sudoku using the TextArena OpenEnv environment."
    )
    parser.add_argument("--model", type=str, default="openenv/Qwen3-1.7B-Sudoku-SFT", help="Model to train.")
    parser.add_argument("--env-url", type=str, required=True, help="URL of your copy of the openenv/sudoku Space.")
    parser.add_argument("--max-steps", type=int, default=75, help="Number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to store outputs and checkpoints.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the trained model to the Hub.")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate", help="vLLM execution mode.")
    parser.add_argument(
        "--vllm-server-url", type=str, default="http://localhost:8000", help="vLLM server URL (server mode only)."
    )
    return parser.parse_args()


prompt = """You are playing Sudoku. Fill every empty cell (shown as '.') of the 9x9 grid with a digit from 1 to 9 so that \
each row, each column and each 3x3 box contains every digit exactly once.

Use the tool `place` to fill one cell at a time. Rows and columns are numbered from 1 to 9. A digit that is not the \
solution of that cell is a mistake. Two mistakes in a row end the game.

"""


def reward_solved(environments, **kwargs) -> list[float]:
    return [1.0 if env.solved else 0.0 for env in environments]


# Partial credit for the cells filled, whether the game ended on mistakes or the model stopped calling the tool. With
# the binary reward alone, a group where no game is solved gives no signal.
def reward_progress(environments, **kwargs) -> list[float]:
    return [env.filled / env.empty for env in environments]


def main() -> None:
    args = parse_args()

    env_url = args.env_url
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else None

    class SudokuEnv:
        def __init__(self):
            self.client = TextArenaEnv(base_url=env_url, headers=headers).sync()

        def reset(self, **kwargs) -> str | None:
            # One connection per game, so the Space's periodic WebSocket drops don't land mid-game, see
            # https://huggingface.co/docs/trl/openenv#long-runs-against-a-space
            self.client.close()
            self.client = TextArenaEnv(base_url=env_url, headers=headers).sync()
            result = self.client.reset()
            # The game returns the whole log each turn, so we store the previous full response and slice out only the
            # newly appended part.
            self._last_full = result.observation.messages[0].content
            board = self._last_full.split("Game Board:\n")[-1]
            self.empty = board.count(".")
            self.filled = 0
            self.solved = False
            self.done = False
            return board

        def place(self, row: int, col: int, number: int) -> str:
            """
            Place a number in an empty cell of the Sudoku grid.

            Args:
                row: Row number (1-9).
                col: Column number (1-9).
                number: Number to place (1-9).

            Returns:
                The updated grid, or the reason the move was rejected.
            """
            if self.done:
                raise ValueError("Game over.")
            result = self.client.step(TextArenaAction(message=f"[{row} {col} {number}]"))
            full = result.observation.messages[0].content
            feedback, self._last_full = full[len(self._last_full) :], full
            self.done = result.done
            if "invalid" in feedback.lower():
                return feedback.split("[GAME]")[-1].strip()
            self.filled += 1
            self.solved = result.done and result.reward >= 1.0
            return feedback.split("Board state: \n")[-1]

    output_dir = args.output_dir or f"{args.model.split('/')[-1]}-GRPO"
    # 4 prompts x 16 games per step
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * (args.max_steps * 4)})

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_solved, reward_progress],
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
            # One game per forward pass: games can reach 16k tokens, since every move returns the whole grid.
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,
            gradient_checkpointing=True,
            bf16=True,
            chat_template_kwargs={"enable_thinking": True},
            max_completion_length=16384,
            mask_truncated_completions=True,
            # Multi-turn rollouts: clip per-token ratios instead of masking whole sequences.
            vllm_importance_sampling_mode="token_truncate",
            # One call per empty cell, plus the moves rejected along the way.
            max_tool_calling_iterations=40,
            save_steps=25,
            save_total_limit=1,
            push_to_hub=args.push_to_hub,
        ),
        environment_factory=SudokuEnv,
    )
    trainer.train()
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
