# On-policy distillation of a coding agent

This example distills [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) into [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) as a coding agent, with [`DistillationTrainer`](https://huggingface.co/docs/trl/distillation_trainer) and `environment_factory`. The student solves [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) problems in a live Python session from OpenEnv's [coding environment](https://github.com/openenv/OpenEnv/tree/main/envs/coding_env): it defines the function, runs it, reads the error, fixes it, and tests it again, over several turns. The teacher grades every token the student writes along the way. In 100 steps, the student goes from 43.6% to 51.4% of the MBPP test cases passed, against 68.2% for the teacher.

## Why an environment

Most distillation recipes work on single answers: a prompt goes in, the student writes one completion, the teacher scores it. An agent's answer is an episode instead. What the student writes on its third turn depends on what the interpreter returned on its second, and that depends on the code the student wrote first. A dataset of the teacher's traces can't follow the student into its own mistakes. An environment can: every episode runs against the student's own session, and the next turn is conditioned on what actually happened.

`environment_factory` gives each episode its own instance of a small class, `CodingSession`. `reset()` opens a fresh Python session on the coding environment server, the public method `run_python(code)` becomes the student's only tool, and the session keeps whatever the student defines between calls. An episode ends when the student answers without calling the tool, after six tool calls (`max_tool_calling_iterations=6`), or when it fills its token budget.

## What the teacher scores

On-policy distillation trains the student on the teacher's per-token distribution over the student's own samples. Here the sample is the whole episode: the student's turns and the tool results between them, concatenated token by token exactly as the student generated them. The teacher is run on those same tokens, so it scores each student token with the full episode so far as context, interpreter output included, and the loss compares the two distributions token by token. The tool results are context only: they are masked out of the loss, since the student did not write them. In this setup they are about four fifths of the episode, mostly tracebacks, which is why `max_completion_length` (the budget for the whole episode) is 4096.

Distillation needs no reward. `CodingSession` still defines `get_reward`, which runs the MBPP tests in the session the student leaves behind. The trainer logs it as `rewards/CodingSession/mean` and never uses it in the loss. It answers the question the loss can't: does the student end the episode with a working function?

## Run it

```sh
python examples/distillation_coding_env/distillation_coding_env.py
```

The script starts the coding environment server on the same machine, in its own process and without the tokens of the training process, since it runs code written by the model. It allows one Python session per episode, 32 per step. On [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs):

```sh
hf jobs uv run --flavor a100-large --timeout 2h --secrets HF_TOKEN \
    examples/distillation_coding_env/distillation_coding_env.py --push-to-hub
```

One A100 holds the student, the teacher and vLLM (colocated, 25% of the memory). Each step runs 32 episodes (8 per device x 4 accumulation steps), and the 100 steps take about 80 minutes. MBPP's train split has 374 problems, so the run goes over each of them about nine times.

## Evaluate

[`evaluate.py`](evaluate.py) runs a model on the 500 problems of the MBPP test split with the same prompt, tool, turn limit and episode budget as training, against its own local coding environment server, and reports the share of the MBPP test cases passed and of the problems fully solved:

```sh
hf jobs uv run --flavor a100-large --timeout 90m examples/distillation_coding_env/evaluate.py \
    --model <you>/Qwen3-1.7B-Distillation-Coding
```

It samples with temperature 0.7, top-p 0.8 and top-k 20 (Qwen3's settings without thinking), two episodes per problem with different seeds.

## Results

![Distillation loss](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/distillation_coding_env/loss.png)

![Tool calls per episode and share of tool-result tokens](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/distillation_coding_env/episodes.png)

| | Steps 1-10 | Steps 91-100 |
| --- | --- | --- |
| Distillation loss | 0.169 | 0.100 |
| Tool calls per episode | 2.2 | 2.0 |
| Tool-result tokens (masked) | 79% | 79% |
| Student tokens per episode | 305 | 283 |
| `get_reward` (MBPP train tests passed, logged only) | 0.55 | 0.62 |

The loss drops fast in the first 20 steps and keeps going down slowly after that. The student calls the interpreter about twice per episode from the first step, and none of its calls fail to run.

On the MBPP test split, two episodes per problem:

| Model | Test cases passed | Problems solved |
| --- | --- | --- |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 43.6% (42.9, 44.3) | 36.7% (36.2, 37.2) |
| [Qwen3-1.7B-Distillation-Coding](https://huggingface.co/sergiopaniego/Qwen3-1.7B-Distillation-Coding) | 51.4% (50.9, 51.9) | 44.3% (44.0, 44.6) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (teacher) | 68.2% (68.1, 68.3) | 61.0% (60.6, 61.4) |

The student gains about 8 points on problems it never trained on, well above the gap between the two seeds, and closes about a third of the distance to the teacher. It also calls the tool less, closer to the teacher: 2.2 calls per episode, against 3.0 for the base model and 1.8 for the teacher. One teacher episode per seed lost its session to the environment server and counts as failed. The training curves are on [Trackio](https://huggingface.co/spaces/sergiopaniego/distillation-coding-env-trackio).

## Whole episodes against the first turn

Distilling whole episodes pays off when the later turns decide the answer. On data-analysis tasks the agent loads files, reads errors and tries again before it answers. We distilled [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) from [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) on [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) tasks served by OpenEnv's `data_analysis_env`, with up to 10 turns per episode and 28 steps of 32 tasks. Each run used the same tasks, once on whole episodes and once on the first turn only (`max_tool_calling_iterations=0`, so no tool runs during training), with three seeds. The base model and all six students were evaluated in the same job, with 8 runs per task:

| | `eval_verified` (61 tasks, 16 turns) | `test` (194 tasks, 10 turns) |
| --- | --- | --- |
| Qwen3.5-2B | 0.189 | 0.143 |
| First turn only, 3 seeds | 0.285 to 0.287 | 0.222 to 0.239 |
| Whole episodes, 3 seeds | 0.344 to 0.355 | 0.273 to 0.282 |
| Whole episodes minus first turn, same seed | +0.059 to +0.068 | +0.042 to +0.059 |

The scores are the share of tasks answered correctly. For every seed, the paired difference between whole episodes and the first turn has its 95% bootstrap interval above 0 on both sets. The students trained on whole episodes explore longer before they answer, so their lead grows when the evaluation allows more turns. The six models are on the Hub ([whole episodes](https://huggingface.co/training-agents-course/qwen3.5-2b-distill-data-episodes-s42), [first turn](https://huggingface.co/training-agents-course/qwen3.5-2b-distill-data-firstturn-s42), seeds 42 to 44), and the training curves are on [Trackio](https://huggingface.co/spaces/training-agents-course/distill-data-analysis-trackio).
