# Teach Qwen3-1.7B to play Sudoku with GRPO

[`grpo_sudoku.py`](grpo_sudoku.py) trains Qwen3-1.7B to solve Sudoku against the [OpenEnv](https://huggingface.co/docs/trl/openenv) TextArena environment, filling one cell per `place(row, col, number)` tool call. [`grpo_sudoku.ipynb`](grpo_sudoku.ipynb) walks through the same steps.

## 1. Can Qwen3-1.7B play Sudoku?

On 30 fixed puzzles with 16 empty cells, with greedy decoding, [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) solves none and fills 2% of the cells, and at 26 empty cells it fills 0.1%. In 20 of the 30 games its first move goes on a cell that is already given. With almost no correct moves, GRPO has very little to compare.

## 2. Warm it up with SFT

A solver plays 1,000 games and writes them as tool-calling conversations ([`openenv/sudoku-tool-calls-sft`](https://huggingface.co/datasets/openenv/sudoku-tool-calls-sft)). It only fills a cell when one digit fits, so every step of its reasoning can be checked on the grid. One epoch of SFT on those games takes 27 minutes on one A100. The SFT model ([`openenv/Qwen3-1.7B-Sudoku-SFT`](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT)) solves 19 of 30 puzzles at 16 empty cells and 10 at 26, filling about half the grid.

This example starts from the published SFT model. To train your own, follow the SFT steps of the [OpenEnv Sudoku tutorial](https://huggingface.co/docs/openenv/tutorials/sudoku-grpo#4-generate-the-sft-data), with [`generate_sft_data.py`](https://github.com/openenv/OpenEnv/blob/main/examples/sudoku/generate_sft_data.py) and [`sft.py`](https://github.com/openenv/OpenEnv/blob/main/examples/sudoku/sft.py), then pass your model with `--model`.

## 3. Train it with GRPO against the environment

GRPO trains at 26 empty cells, where the SFT model still has room to improve and every group has games that go further than others. The reward has two terms: 1 for a solved puzzle, and the fraction of empty cells filled, so that a group where no game is solved still gives a signal.

Each step plays 64 games, one environment session each. [Duplicate the `openenv/sudoku` Space](https://huggingface.co/spaces/openenv/sudoku?duplicate=true) and set these variables on your copy:

| Variable | Value | Why |
| --- | --- | --- |
| `TEXTARENA_KW_CLUES` | `55` | About 26 empty cells. The Space default is 60 clues, about 21. |
| `MAX_CONCURRENT_ENVS` | `96` | 64 games at once, plus room for the sessions being closed between games. |

```sh
hf jobs uv run --flavor a100-large --timeout 18h --secrets HF_TOKEN examples/grpo_sudoku/grpo_sudoku.py \
    --env-url https://<user>-sudoku.hf.space --push-to-hub
```

The script's docstring has the commands for a local GPU, with vLLM colocated or as a separate server. Our run did 75 steps on one A100 80GB, in about 15 hours, and steps get slower as the model improves, since solved games are longer. The script opens a new connection to the Space for every game, because a Space closes its WebSockets about every 2 hours. [OpenEnv integration](https://huggingface.co/docs/trl/openenv) covers the rewards and the reconnects in detail.

![Puzzles solved and cells filled during GRPO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/sudoku-grpo-curve.png)

The training curves are on [Trackio](https://huggingface.co/spaces/openenv/sudoku-runs).

## 4. Measure again

![Base model, SFT and SFT + GRPO playing the same puzzles](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/sudoku-progress.gif)

Puzzles 7 to 12 of [`puzzles_c55.json`](https://github.com/openenv/OpenEnv/blob/main/examples/sudoku/puzzles_c55.json) (26 empty cells), where the SFT model solves 2 and SFT + GRPO 4, close to their rates over all 30.

| Model | 26 empty cells: solved | cells filled | 16 empty cells: solved | cells filled |
| --- | --- | --- | --- | --- |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 0/30 | 0.1% | 0/30 | 1.9% |
| + SFT ([model](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT)) | 10/30 | 55.8% | 19/30 | 79.1% |
| + SFT + GRPO ([model](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT-GRPO)) | 22/30 | 83.2% | 26/30 | 87.5% |

Greedy decoding on the same 30 puzzles at each level, with up to 40 moves per game. GRPO more than doubles the puzzles solved at 26 empty cells, the level it trained on. Thirty puzzles is a small sample, and greedy decoding on a different GPU changes a few games, so expect the same ordering in a new run rather than the exact numbers.

## The whole pipeline

The puzzles, the SFT data, the SFT run, the evaluation and the GIF are explained step by step in the [OpenEnv Sudoku tutorial](https://huggingface.co/docs/openenv/tutorials/sudoku-grpo), with the scripts in [`examples/sudoku`](https://github.com/openenv/OpenEnv/tree/main/examples/sudoku) of the OpenEnv repo. The whole pipeline costs about $42 on Hugging Face Jobs.
