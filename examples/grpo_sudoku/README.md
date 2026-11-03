# Sudoku: GRPO against an OpenEnv environment

[`grpo_sudoku.py`](grpo_sudoku.py) trains Qwen3-1.7B to solve Sudoku against the [OpenEnv](https://huggingface.co/docs/trl/openenv) TextArena environment, filling one cell per `place(row, col, number)` tool call. It starts from an SFT warmup on solver-played games ([`openenv/Qwen3-1.7B-Sudoku-SFT`](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT)), since the base model solves no puzzle and leaves GRPO nothing to learn from. [`grpo_sudoku.ipynb`](grpo_sudoku.ipynb) walks through the same steps.

![Base model, SFT and SFT + GRPO playing the same puzzles](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/sudoku-progress.gif)

| Model | 26 empty cells: solved | cells filled | 16 empty cells: solved | cells filled |
| --- | --- | --- | --- | --- |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 0/30 | 0.1% | 0/30 | 1.9% |
| + SFT ([model](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT)) | 10/30 | 55.8% | 19/30 | 79.1% |
| + SFT + GRPO ([model](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT-GRPO)) | 22/30 | 83.2% | 26/30 | 87.5% |

Greedy decoding on the same 30 puzzles at each level. GRPO ran 75 steps on one A100 80GB, in about 15 hours. The training curves are on [Trackio](https://huggingface.co/spaces/openenv/sudoku-runs).

![Puzzles solved and cells filled during GRPO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/sudoku-grpo-curve.png)

## SFT warmup

This example starts from the published SFT model. To train your own, follow the SFT steps of the [OpenEnv Sudoku tutorial](https://huggingface.co/docs/openenv/tutorials/sudoku-grpo#4-generate-the-sft-data). They use two scripts: [`generate_sft_data.py`](https://github.com/openenv/OpenEnv/blob/main/examples/sudoku/generate_sft_data.py) plays 1,000 games with a solver and writes them as tool-calling conversations ([`openenv/sudoku-tool-calls-sft`](https://huggingface.co/datasets/openenv/sudoku-tool-calls-sft)), and [`sft.py`](https://github.com/openenv/OpenEnv/blob/main/examples/sudoku/sft.py) trains on them with `SFTTrainer`, in 27 minutes on one A100. Then pass your model with `--model`.

## Run GRPO

Each step plays 64 games, one environment session each. [Duplicate the `openenv/sudoku` Space](https://huggingface.co/spaces/openenv/sudoku?duplicate=true) and set these variables on your copy:

| Variable | Value | Why |
| --- | --- | --- |
| `TEXTARENA_KW_CLUES` | `55` | About 26 empty cells. The Space default is 60 clues, about 21. |
| `MAX_CONCURRENT_ENVS` | `96` | 64 games at once, plus room for the sessions being closed between games. |

```sh
hf jobs uv run --flavor a100-large --timeout 18h --secrets HF_TOKEN examples/grpo_sudoku/grpo_sudoku.py \
    --env-url https://<user>-sudoku.hf.space --push-to-hub
```

The script's docstring has the commands for a local GPU, with vLLM colocated or as a separate server.

Two rewards: 1 for a solved puzzle, plus the fraction of empty cells filled, so that a group where no game is solved still gives a signal. The script opens a new connection to the Space for every game, because a Space closes WebSockets after about 2 hours. [OpenEnv integration](https://huggingface.co/docs/trl/openenv) covers both in detail.

## The whole pipeline

The puzzles, the SFT data, the SFT run, the evaluation and the GIF are explained step by step in the [OpenEnv Sudoku tutorial](https://huggingface.co/docs/openenv/tutorials/sudoku-grpo), with the scripts in [`examples/sudoku`](https://github.com/openenv/OpenEnv/tree/main/examples/sudoku) of the OpenEnv repo. The whole pipeline costs about $42 on Hugging Face Jobs.
