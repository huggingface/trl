# On-policy distillation of a data-analysis agent

This example distills [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) into [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) as a data-analysis agent, with [`DistillationTrainer`](https://huggingface.co/docs/trl/distillation_trainer) and `environment_factory`. The student answers [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) questions about real CSV, TSV and JSON files from a live Python session of OpenEnv's [data-analysis environment](https://github.com/openenv/OpenEnv/tree/main/envs/data_analysis_env): it loads the files, reads the errors, tries again and submits an answer, over up to 10 turns. The teacher scores every token the student writes along the way.

The same script with `--first-turn` distills the first turn only, and the comparison is the point of the example: on 61 checked tasks, the student goes from 0.19 of the questions answered correctly to about 0.29 with first-turn distillation and about 0.35 with whole episodes, for each of three seeds.

## Why an environment

Most distillation recipes work on single answers: a prompt goes in, the student writes one completion, the teacher scores it. An agent's answer is an episode instead. What the student writes on its third turn depends on what the session returned on its second, and that depends on the code the student wrote first. A dataset of the teacher's traces can't follow the student into its own mistakes. An environment can: every episode runs against the student's own session, and the next turn is conditioned on what actually happened.

`environment_factory` gives each episode its own instance of a small class, `DataAnalysisSession`. `reset()` starts the row's task in a fresh session on the environment server, and the public methods `run_python(code)` and `submit_answer(answer)` become the student's tools. An episode ends when the student submits an answer, replies without calling a tool, after nine rounds of tool calls (`max_tool_calling_iterations=9`, so the student writes at most 10 turns), or when it fills its token budget. When one turn is left, the tool result says so.

## What the teacher scores

On-policy distillation trains the student on the teacher's per-token distribution over the student's own samples. Here the sample is the whole episode: the student's turns and the tool results between them, concatenated token by token exactly as the student generated them. The teacher is run on those same tokens, so it scores each student token with the full episode so far as context, data previews and tracebacks included, and the loss compares the two distributions token by token. The tool results are context only: they are masked out of the loss, since the student did not write them. They still count toward `max_completion_length`, the budget for the whole episode, which is 8192 here.

With `--first-turn`, no tool runs (`max_tool_calling_iterations=0`): the student writes its first turn, usually a first `run_python` call, and the teacher scores that turn alone. Everything else is the same: the tasks, their order, the seed and the number of steps.

Distillation needs no reward. `DataAnalysisSession` still defines `get_reward`, the environment's grade of the answer sent with `submit_answer` (1 or 0, with the dataset's own grader). The trainer logs it as `rewards/DataAnalysisSession/mean` and never uses it in the loss. It stays at 0 with `--first-turn`, where no tool runs.

## Run it

The script starts the environment server on the same machine, in its own process. The server runs code written by the model, so it gets an environment built from constants, without the tokens of the training process, and runs each session as its own unprivileged user. That needs root on Linux, which is what a container or [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) gives, and the script checks it from a session before training. It also needs [`uv`](https://docs.astral.sh/uv/), to install the server in its own virtual environment.

```sh
python examples/distillation_data_analysis_env/distillation_data_analysis_env.py
python examples/distillation_data_analysis_env/distillation_data_analysis_env.py --first-turn
```

On Hugging Face Jobs, with a token only to push the model:

```sh
hf jobs uv run --flavor a100-large --timeout 2h --secrets HF_TOKEN \
    examples/distillation_data_analysis_env/distillation_data_analysis_env.py --push-to-hub
```

One A100 holds the student, the teacher and vLLM (colocated, 25% of the memory). Each step runs 32 tasks (4 per device x 8 accumulation steps), and the 28 steps go about once over the 885 training tasks: SmolDataEnvs `train` without the Kaggle datasets of its `eval` split, at most six tasks per Kaggle dataset. Whole episodes take about an hour of training, the first turn about 11 minutes.

## Evaluate

[`evaluate.py`](evaluate.py) runs the base model and the given models on the same tasks in one job, with the same prompt, tools, tool-result cap and last-turn reminder as training, against its own local environment server. Each model is served with vLLM and runs every task `--num-samples` times (8 by default). It reports each model's share of correct answers with a 95% bootstrap interval over tasks, and the paired difference between every two models, task by task:

```sh
hf jobs uv run --flavor a100-large --timeout 2h examples/distillation_data_analysis_env/evaluate.py \
    --models first-turn=<your-username>/Qwen3.5-2B-Distillation-Data-First-Turn \
    episodes=<your-username>/Qwen3.5-2B-Distillation-Data-Episodes
```

By default it runs the 61 tasks of SmolDataEnvs `eval_verified` (checked answers, no data file shared with `train`). `--split test` runs the 194 harder `test` tasks, and `--max-turns` changes the turn limit.

## Results

We trained both versions with three seeds, then evaluated the base model and the six students in the same job, with 8 runs per task. Scores are the share of correct answers, and the brackets hold the 95% bootstrap interval over tasks of the paired difference between whole episodes and the first turn of the same seed:

| | `eval_verified`, 61 tasks, 16 turns | `test`, 194 tasks, 10 turns |
| --- | --- | --- |
| Qwen3.5-2B | 0.189 | 0.143 |
| First turn, seeds 42, 43, 44 | 0.287, 0.285, 0.287 | 0.239, 0.227, 0.222 |
| Whole episodes, seeds 42, 43, 44 | 0.355, 0.344, 0.350 | 0.281, 0.273, 0.282 |
| Whole episodes minus first turn, seed 42 | +0.068 [+0.033, +0.105] | +0.042 [+0.016, +0.069] |
| Seed 43 | +0.059 [+0.023, +0.096] | +0.046 [+0.021, +0.070] |
| Seed 44 | +0.064 [+0.029, +0.100] | +0.059 [+0.034, +0.084] |

Both kinds of distillation beat the base model, and for every seed whole episodes beat the first turn with the interval above 0 on both task sets. The students trained on whole episodes explore longer before they answer, so their lead grows when the evaluation allows more turns: with 10 turns on the 61 `eval_verified` tasks, it was smaller, +0.04 to +0.05, and one seed's interval touched 0.

On tasks where the first attempt decides most of the score, the later turns have little left to fix. On MBPP coding problems, with Qwen3-1.7B learning from Qwen3-8B over up to six rounds of tool calls in OpenEnv's coding environment, whole episodes and the first turn tied in our runs: -0.007 [-0.036, +0.020] on 257 MBPP test problems. That is why this example uses data analysis.

The six models are on the Hub: whole episodes ([seed 42](https://huggingface.co/training-agents-course/qwen3.5-2b-distill-data-episodes-s42), [43](https://huggingface.co/training-agents-course/qwen3.5-2b-distill-data-episodes-s43), [44](https://huggingface.co/training-agents-course/qwen3.5-2b-distill-data-episodes-s44)) and first turn ([seed 42](https://huggingface.co/training-agents-course/qwen3.5-2b-distill-data-firstturn-s42), [43](https://huggingface.co/training-agents-course/qwen3.5-2b-distill-data-firstturn-s43), [44](https://huggingface.co/training-agents-course/qwen3.5-2b-distill-data-firstturn-s44)). The training curves are on [Trackio](https://huggingface.co/spaces/training-agents-course/distill-data-analysis-trackio). Those runs come from an earlier version of this script: their logged reward is named `rewards/DataEnv/mean` and also counts answers given in a final reply. This script logs `rewards/DataAnalysisSession/mean`, which only counts answers sent with `submit_answer`. `evaluate.py` also reads an answer from a final reply without a tool call, so the evaluation numbers above count both.
