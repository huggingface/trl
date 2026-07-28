# Harbor Integration for Training LLMs with Environments

[Harbor](https://www.harborframework.com) is a framework for running agentic tasks in sandboxes. It decouples a **task** (instruction + sandbox image + verifier), a **harness/agent** (the tool surface + loop), and a **sandbox** (`docker`, `e2b`, `daytona`, `gke`, …) so they can be mixed freely. This makes it a natural fit for RL: the same task suite can be trained with different tool surfaces, on whichever sandbox backend you prefer.

This guide covers **how to integrate Harbor with TRL**. For Harbor itself, see the [Harbor docs](https://www.harborframework.com/docs).

> [!NOTE]
> The integration lives at `trl.experimental.harbor` and is gated behind the `trl[harbor]` extra (lazy-imported — non-users pay nothing).

## When to use Harbor environments

[`GRPOTrainer`] supports environment-based training via the `environment_factory` slot — see [OpenEnv](openenv) for the general contract. Use Harbor when you want to train against a **Harbor task suite**: a directory tree of tasks, each a self-contained sandbox + verifier (for example, a data-analysis agent suite where the model explores files in a sandbox and writes an answer that a grader checks).

## Installation

```bash
pip install trl[harbor]
```

> [!IMPORTANT]
> Harbor drives generation through vLLM and uses `environment_factory`, which requires `vllm>=0.22.0` and `transformers>=5.2.0`.
>
> ```bash
> pip install 'vllm>=0.22.0'
> ```

This installs the `harbor` framework (Python >= 3.12). The integration imports `harbor` lazily and runs it **in-process**, so users who don't touch `trl.experimental.harbor` aren't affected.

A sandbox backend must also be installed and reachable at train time. Harbor keeps cloud backends behind its own extras, so install the one you intend to use and provide its credentials:

```bash
pip install "harbor[e2b]"      # E2B cloud sandbox  -> environment_type="e2b",  needs E2B_API_KEY
# docker backend (environment_type="docker", Harbor's default) just needs a reachable Docker daemon
```

## Quick start

`HarborSpec` wires a single Harbor task suite into the three TRL trainer slots — `train_dataset`, `environment_factory`, `reward_funcs` — by exposing properties that map 1:1 to those kwarg names:

```python
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.harbor import HarborSpec

spec = HarborSpec("AdithyaSK/data_agent_rl_environment_train", agent="bash", num_tasks=64)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=GRPOConfig(
        num_generations=8,
        max_steps=50,
        max_tool_calling_iterations=25,
        log_completions=True,
    ),
    train_dataset=spec.train_dataset,
    environment_factory=spec.environment_factory,
    reward_funcs=spec.reward_funcs,
)
trainer.train()
```

Under the hood `HarborSpec` does three things, lazily on first access:

1. **`spec.train_dataset`**: resolves the task suite to local task directories (downloading the HF dataset if needed) and builds a `datasets.Dataset` with `prompt` (empty — the env's instruction is appended at `reset`), `task_dir`, `task_index`, plus per-task `task.toml` metadata columns.
2. **`spec.environment_factory`**: returns a zero-arg callable producing a fresh per-rollout [`~trl.experimental.harbor.HarborEnv`]. On `reset(task_dir)` it starts the task's Harbor sandbox and returns its instruction; tool methods exec into the sandbox; `env.reward` runs the verifier once after the rollout.
3. **`spec.reward_funcs`**: an outcome reward that reads the Harbor verifier's scalar per rollout.

## The dataset

`dataset` is either a Hugging Face dataset repo id holding a Harbor task tree, or a local path containing a `tasks/` subtree. Each task is a directory:

```
tasks/<task_id>/
├── instruction.md          # the task prompt (returned by reset)
├── task.toml               # config + metadata (gold answer, difficulty, ...)
├── environment/            # Dockerfile (+ any pre-agent data hooks)
└── tests/                  # test.sh / grader → writes the reward
```

Select a subset with `num_tasks` or `indices` (mutually exclusive):

```python
spec = HarborSpec("AdithyaSK/data_agent_rl_environment_train", num_tasks=10)        # first 10
spec = HarborSpec("AdithyaSK/data_agent_rl_environment_train", indices=[0, 5, 13])  # specific
```

## Agents: external vs installed

Harbor supports two ways an agent drives a task, and the distinction determines what can be trained with RL:

- [**External agents**](https://www.harborframework.com/docs/agents#external-agents) run *outside* the sandbox and drive the loop themselves, issuing commands into the container through Harbor's environment interface ("typically by executing bash commands via the `exec` method"). The agent decides each action and interprets each result; the sandbox only executes.
- [**Installed agents**](https://www.harborframework.com/docs/agents#installed-agents) are installed *into the container image* and run there as a headless subprocess (extending `BaseInstalledAgent`). Harbor launches the agent inside the sandbox and parses its trajectory file afterward (`populate_context_post_run`); the agent runs autonomously with its own inference.

**TRL's integration is the external-agent pattern, and only that pattern is supported for now.** RL training requires the trainer to drive the rollout turn by turn: the *policy model being trained* generates each turn, and TRL captures its tokens and log-probs and applies the environment mask — exactly what `environment_factory` provides over a black-box `rollout_func`. An installed agent is opaque to this: it runs inside the container with its *own* model and only emits a trajectory after the fact, so there are no policy tokens or log-probs for the trainer to optimize, and the model under training is never invoked. A [`~trl.experimental.harbor.HarborEnv`] is therefore an external agent — its tool methods `exec` into the sandbox, but the loop, and the model under training, stay in TRL.

## Selecting the base agent (harness)

The **base agent** is the harness — which tool methods the env exposes and how it submits. Select it with `agent=`:

```python
HarborSpec(dataset, agent="bash")                          # built-in single-bash-tool harness
HarborSpec(dataset, agent="my_pkg.harnesses:JupyterEnv")   # import path to your HarborEnv subclass
HarborSpec(dataset, agent="path/to/harness.py:JupyterEnv") # file path to your HarborEnv subclass
HarborSpec(dataset, agent=MyHarborEnv)                     # a HarborEnv subclass directly
```

The built-in `"bash"` harness ([`~trl.experimental.harbor.HarborBashEnv`]) exposes one `bash` tool and submits by writing `/workdir/answer.txt`. Two richer harnesses ship as examples — each in its own folder with a README listing its tools — under [`examples/scripts/harbor/harnesses/`](https://github.com/huggingface/trl/tree/main/examples/scripts/harbor/harnesses):

- [`jupyter/`](https://github.com/huggingface/trl/tree/main/examples/scripts/harbor/harnesses/jupyter) (`JupyterEnv`) — a stateful Python kernel (variables persist across cells) + a shell tool.
- [`terminal_notes/`](https://github.com/huggingface/trl/tree/main/examples/scripts/harbor/harnesses/terminal_notes) (`TerminalNotesEnv`) — 6 shell tools (incl. background processes) + a 4-tool persistent note toolkit.

```python
HarborSpec(dataset, agent="examples/scripts/harbor/harnesses/jupyter/env.py:JupyterEnv")
HarborSpec(dataset, agent="examples/scripts/harbor/harnesses/terminal_notes/env.py:TerminalNotesEnv")
```

To write your own harness, subclass [`~trl.experimental.harbor.HarborEnv`] and add tool methods — every public method becomes a tool (TRL discovers them with `inspect.getmembers`), so give each a typed signature and a docstring (used to build the tool schema). Keep helpers underscore-prefixed. Use `self._exec(cmd)` to run shell commands in the sandbox, and set `PROMPT_SUFFIX` to append harness guidance to the task instruction:

```python
from trl.experimental.harbor import HarborEnv

class GrepEnv(HarborEnv):
    PROMPT_SUFFIX = "\n\nUse `grep` and `read_file`. Submit by writing /workdir/answer.txt."

    def grep(self, pattern: str, path: str) -> str:
        """Search for `pattern` under `path`.

        Args:
            pattern: The regex to search for.
            path: The file or directory to search.
        """
        return self._exec(f"grep -rn {pattern!r} {path!r}")
```

## The sandbox backend

`environment_type` is passed straight through to Harbor (not validated by TRL):

```python
HarborSpec(dataset, environment_type="e2b")   # cloud sandbox (offloads provisioning), needs E2B_API_KEY
HarborSpec(dataset, environment_type="docker")  # default; needs a local Docker daemon
```

`e2b` is recommended for cluster training: only `environment.exec` crosses into the cloud sandbox, so the GPUs stay dedicated to the policy and you can run many rollouts concurrently.

## Reward functions

`spec.reward_funcs` defaults to an outcome reward — per rollout it reads the Harbor verifier's scalar (`env.reward`), computed once after the rollout by running the task's `tests/` verifier in the sandbox. For a custom reward, write a regular TRL reward function:

```python
def my_reward(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
```

## API

[[autodoc]] trl.experimental.harbor.HarborSpec

[[autodoc]] trl.experimental.harbor.HarborEnv

[[autodoc]] trl.experimental.harbor.HarborBashEnv

## Limitations

- The integration is in `trl.experimental` — APIs may change. Set `TRL_EXPERIMENTAL_SILENCE=1` to silence the warning in CI logs.
- Harbor's async sandbox client is bound to one event loop, so each env drives start/exec/verify synchronously on its own loop; sandbox provisioning is therefore sequential across the generation batch (cloud backends like `e2b` mitigate the per-sandbox cost).
- A single `HarborSpec` covers one task suite + one harness; multi-suite training is not supported yet.

## Reference

- [Harbor framework](https://www.harborframework.com)
- [Harbor RL training docs](https://www.harborframework.com/docs/training-workflows/rl)
