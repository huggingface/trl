# Wordle: GRPO against an OpenEnv environment

[`grpo_wordle.py`](grpo_wordle.py) trains Qwen3-1.7B to play [Wordle](https://en.wikipedia.org/wiki/Wordle) against the [OpenEnv](https://huggingface.co/docs/trl/openenv) TextArena environment, with one `guess` tool call per word. It starts from an SFT warmup ([`openenv/Qwen3-1.7B-Wordle-SFT`](https://huggingface.co/openenv/Qwen3-1.7B-Wordle-SFT)), since the base model never wins and leaves GRPO nothing to learn from. [`grpo_wordle.ipynb`](grpo_wordle.ipynb) walks through the same steps.

![Base model, SFT and SFT + GRPO playing the same 30 words](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/wordle-progress.gif)

| Model | Wins on 30 words | Guesses per win | Invalid guesses |
| --- | --- | --- | --- |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 0/30 | - | 24 |
| + SFT ([model](https://huggingface.co/openenv/Qwen3-1.7B-Wordle-SFT)) | 10/30 | 3.2 | 41 |
| + SFT + GRPO ([model](https://huggingface.co/openenv/Qwen3-1.7B-Wordle-SFT-GRPO)) | 20/30 | 4.1 | 3 |

Greedy decoding on the same 30 words. GRPO ran 150 steps on one A100 80GB, in about 2 hours 15 minutes. The training curves are on [Trackio](https://huggingface.co/spaces/openenv/wordle-runs), and the [Wordle Arena](https://huggingface.co/spaces/openenv/wordle-arena) Space replays the three models on any of the 30 words. Our run allowed 6 tool rounds per game, and the script now allows 12, so that invalid guesses don't use up the 6 legal ones.

## SFT warmup

This example starts from the published SFT model. To train your own, the OpenEnv repo has the two scripts: [`generate_sft_data.py`](https://github.com/openenv/OpenEnv/blob/main/examples/wordle/generate_sft_data.py) plays 1,000 games with a solver and writes them as tool-calling conversations ([`openenv/wordle-tool-calls-sft`](https://huggingface.co/datasets/openenv/wordle-tool-calls-sft)), and [`sft.py`](https://github.com/openenv/OpenEnv/blob/main/examples/wordle/sft.py) trains on them with `SFTTrainer`, in 7 minutes on one A100. Then pass your model with `--model`.

## Run GRPO

Each step plays 64 games, one environment session each, and the shared [`openenv/wordle`](https://huggingface.co/spaces/openenv/wordle) Space allows 8. [Duplicate it](https://huggingface.co/spaces/openenv/wordle?duplicate=true), set the `MAX_CONCURRENT_ENVS=64` variable on your copy, and pass its URL:

```sh
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN examples/grpo_wordle/grpo_wordle.py \
    --env-url https://<user>-wordle.hf.space --push-to-hub
```

The script's docstring has the commands for a local GPU, with vLLM colocated or as a separate server.

The rewards are three separate terms (a win, paid more the faster it comes, partial credit from the best guess, and a penalty per invalid move), so TRL logs each one. [OpenEnv integration](https://huggingface.co/docs/trl/openenv) covers the environment class and the rewards in detail.

## The whole pipeline

The SFT data, the SFT run, the evaluation and the GIF are explained step by step in the [OpenEnv Wordle tutorial](https://huggingface.co/docs/openenv/tutorials/wordle-grpo), with the scripts in [`examples/wordle`](https://github.com/openenv/OpenEnv/tree/main/examples/wordle) of the OpenEnv repo. The whole pipeline costs about $6 on Hugging Face Jobs.
