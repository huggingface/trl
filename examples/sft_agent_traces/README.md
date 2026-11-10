# Teach a small model to work as an agent with SFT on agent traces

A 2B model dropped into a coding agent usually loops: it calls tools for twenty turns, runs out of time, and often
never writes an answer. A much larger model in the same agent solves most tasks in about six turns. This example
records the large model working through real tasks, keeps the conversations that ended with a correct answer, and
fine-tunes the small model on them with [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer). It is the first
stage of an agentic post-training recipe: SFT on traces gives the student a working agent loop, which RL then
improves.

| Step | Script | Where it runs |
| --- | --- | --- |
| 1. Run the teacher on training tasks, keep every rollout | [`rollouts.py`](rollouts.py) | CPU Job, the teacher on Inference Providers |
| 2. Keep the solved rollouts as SFT rows | [`build_dataset.py`](build_dataset.py) | anywhere |
| 3. Fine-tune the student | [`sft.py`](sft.py) | one GPU |
| 4. Evaluate the student before and after | [`rollouts.py`](rollouts.py) `--serve` | one GPU, vLLM |

Every script takes the number of tasks or rows as a flag, so the same commands run a 10-task smoke test or the whole
split.

## 1. The tasks

[SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-train) are data-analysis questions over
real Kaggle datasets, such as "what percentage of matches ended in a draw?", built from the notebooks of the
[Jupyter Agent dataset](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset). Each task is a
[Harbor](https://harborframework.com/) task: the question, the dataset's files, which the sandbox pulls at start, the
gold answer, and a deterministic grader that scores the agent's `answer.txt` 1 or 0. The train split has 5,000 tasks
and the [eval split](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-eval) 144.

The agent is [OpenCode](https://opencode.ai), an open-source coding agent with `bash`, `read` and other file tools.
[OpenEnv](https://github.com/openenv/OpenEnv) runs it on a task in a
[Hugging Face Sandbox](https://huggingface.co/docs/huggingface_hub/guides/sandboxes) and records every call it makes
to the model: the messages sent, the tools offered and the reply. This record is what we train on.

## 2. Generate traces with a teacher

The teacher is [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), served by
[Inference Providers](https://huggingface.co/docs/inference-providers), so the Job generating traces needs no GPU.
Its license (Apache-2.0) allows training on its outputs, and it shares a tokenizer with the student, which on-policy
distillation needs later on.

```sh
hf jobs uv run --flavor cpu-upgrade --timeout 4h --secrets HF_TOKEN \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/sft_agent_traces/rollouts.py \
    --model qwen/qwen3.5-397b-a17b --llm-url https://router.huggingface.co/novita/v3/openai \
    --dataset FineEnvs/SmolDataEnvs-harbor-train --exclude-datasets-of FineEnvs/SmolDataEnvs-harbor-eval \
    --num-tasks 500 --workers 12 --push-to <you>/smoldataenvs-rollouts --run-name teacher-397b
```

`--exclude-datasets-of` skips the training tasks built on a Kaggle dataset that the eval split also uses. The splits
share no question, but they were drawn task by task, not dataset by dataset: 122 of the 144 eval tasks and 185 of the
250 test tasks use a Kaggle dataset that some training task uses too, and a student trained on those has already seen
their files and columns. The filter keeps 2,879 of the 5,000 training tasks.

The URL names one provider, since the router's default provider for this model rejects a parameter OpenEnv sends.
The teacher solves 78% of the clean training tasks, in a median of 6 turns and about 2 minutes per task. Every result
JSON is uploaded as soon as its task ends, and `--start` and `--num-tasks` pick the next slice, so several Jobs can
split a larger run.

## 3. Build the SFT dataset

```sh
uv run examples/sft_agent_traces/build_dataset.py \
    --rollouts <you>/smoldataenvs-rollouts --runs teacher-397b --push-to <you>/smoldataenvs-agent-traces
```

One row per solved task: the whole conversation of the agent's last call to the model (system prompt, task, every
assistant turn with its tool calls, every tool output) plus the final reply, in the conversational format with a
`tools` column. Two details matter here:

- Tool-call arguments arrive as JSON strings, and the Qwen3.5 chat template iterates over them as a dict, so they
  are parsed.
- Messages and tools are typed as [`Json`](https://huggingface.co/docs/datasets/about_dataset_features). Inferred
  from the rows, every tool call would get one struct with the argument names of every tool, `None` where a call
  doesn't use them, and the chat template would render those.

The teacher's reasoning is not in the rows, since OpenEnv's record of a call keeps the reply's text and tool calls
only. The student trains in no-think mode, which is also the default of Qwen3.5-2B.

## 4. Fine-tune the student

```sh
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/sft_agent_traces/sft.py \
    --dataset <you>/smoldataenvs-agent-traces --push-to <you>/Qwen3.5-2B-SmolDataEnvs-SFT
```

`assistant_only_loss=True` puts the loss on the assistant turns only, so the student learns the teacher's tool calls
and answers, not the tool outputs. Qwen3.5's chat template doesn't mark assistant turns, so TRL swaps in its training
version of the template, which does. A trace is about 10k tokens, mostly OpenCode's system prompt and tool schemas,
and only 4 to 8% of it is trained on. About 5% of the traces run longer than 16k tokens (up to 78k), and `sft.py` drops
them rather than truncating them, since truncation would cut their final answer. About 500 traces train for 3 epochs in
about 16 minutes on one H200.

## 5. Evaluate

The same script runs the student, with `--serve` to serve it with vLLM on the Job's GPU:

```sh
hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/sft_agent_traces/rollouts.py \
    --model <you>/Qwen3.5-2B-SmolDataEnvs-SFT --serve --reward-key correctness,reward \
    --dataset FineEnvs/SmolDataEnvs-harbor-eval --num-tasks 144 --workers 8 \
    --push-to <you>/smoldataenvs-rollouts --run-name eval-sft
```

Every model below is evaluated on the same 307 tasks whose Kaggle dataset no training trace used: the 144 eval tasks
and 163 of the 250 tasks of the [test split](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-test), the
ones whose Kaggle dataset backs no training task outside the eval split's datasets. `--reward-key correctness,reward`
reads whichever reward a task's grader wrote.

| Model | Eval (144) | Test (163) |
| --- | --- | --- |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 7.6% | 5.5% |
| [Qwen3.5-2B-SmolDataEnvs-SFT-500](https://huggingface.co/sergiopaniego/Qwen3.5-2B-SmolDataEnvs-SFT-500) (the commands above) | 33.3% | 34.4% |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) (teacher) | 71.5% | 74.2% |

With about 500 traces the student goes from about 6% to about 34% on tasks built on data it never saw, close to half
of the teacher. It also learns the shape of the work: a median of 8 to 9 turns instead of 18 to 20, and, on the tasks
whose grader records it, an answer on two thirds of them instead of one third. What the student still gets wrong is
mostly looping: a quarter of the tasks hit the task's 10-minute agent timeout (a tenth for the base model, which gives
up sooner, and none for the teacher).

The eval takes longer than the training. Each turn waits for a tool to run in a remote sandbox, and each task starts
and stops its own sandbox, so the GPU idles most of the time. More `--workers` make it faster, up to what your
sandbox quota allows.

### How many traces?

Most of the gain comes from the first few hundred traces. Generating from the whole clean train split
(`--num-tasks 2879`) gives four times as many and adds 3 to 4 points:

| Model | Traces | Eval (144) | Test (163) |
| --- | --- | --- | --- |
| [Qwen3.5-2B-SmolDataEnvs-SFT-500](https://huggingface.co/sergiopaniego/Qwen3.5-2B-SmolDataEnvs-SFT-500) | 473 | 33.3% | 34.4% |
| [Qwen3.5-2B-SmolDataEnvs-SFT-1k](https://huggingface.co/sergiopaniego/Qwen3.5-2B-SmolDataEnvs-SFT-1k) | 963 | 32.6% | 38.7% |
| [Qwen3.5-2B-SmolDataEnvs-SFT-full](https://huggingface.co/sergiopaniego/Qwen3.5-2B-SmolDataEnvs-SFT-full) | 2,238 | 36.8% | 38.7% |

With 144 and 163 tasks, a difference of a few points is within noise. The traces, every raw rollout and the training
curves are in [this collection](https://huggingface.co/collections/sergiopaniego/sft-on-agent-traces-with-trl).

### Why the dataset filter matters

Trained on traces from any training task, without `--exclude-datasets-of`, the student scores 17.3% on the eval tasks
whose Kaggle dataset no trace used and 42.4% on those that share a dataset with some trace: most of the apparent gain
is familiarity with the files.

## Notes

- **Graders.** The eval split mixes two grader versions: one writes a single `reward`, the other `correctness`,
  `submission` and `tool_efficiency`. Both check the answer the same way, and the scripts read `correctness` first
  and `reward` otherwise.
- **Images.** HF Sandboxes need a prebuilt image, which the eval split's `task.toml` files don't pin. Its Dockerfile
  is the train split's, so `rollouts.py` pins the train image.
- **Job names.** Until a `huggingface_hub` release ships huggingface/huggingface_hub#5306, a sandbox on a
  digest-pinned image fails to start, and `rollouts.py` patches the name it derives.
- A rollout that ends right after a tool call, with no answer, usually means a failed call to the provider.
  `build_dataset.py` lists those tasks so you can rerun them with `--task-indices`.
