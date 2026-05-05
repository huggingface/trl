# OpenReward Integration for Training LLMs with Environments

[OpenReward](https://openreward.ai) is an open ecosystem for RL environments built on the [Open Reward Standard (ORS)](https://openrewardstandard.io) — a public, language-agnostic HTTP/SSE protocol for how an environment exposes its tasks, tools, sessions, and rewards. Because ORS is just a protocol, the same environment can run on the [OpenReward platform](https://openreward.ai), self-hosted on any container service, or locally on `localhost` for development. A catalog of ready-to-use environments is available at [openreward.ai](https://openreward.ai).

This guide covers **how to integrate OpenReward with TRL**. For more on the standard itself, see the [ORS docs](https://docs.openreward.ai/).

> [!NOTE]
> The integration lives at `trl.experimental.openreward` and is gated behind the `trl[openreward]` extra (lazy-imported — non-users pay nothing).

## When to use OpenReward environments

[`GRPOTrainer`] supports environment-based training via the `environment_factory` slot — see [OpenEnv](openenv) for the general contract. Use OpenReward when you want to train against an ORS-speaking environment: the [OpenReward catalog](https://openreward.ai) (e.g. `Eigent/SETA`, `kanishk/EndlessTerminals`, `nebius/SWE-rebench-V2`), an env you self-host on your own infra, or a local server you're developing.

## Installation

```bash
pip install trl[openreward]
```

This installs the `openreward` Python SDK. The integration itself imports `openreward` lazily, so users who don't touch `trl.experimental.openreward` aren't affected.

## Quick start

The `OpenRewardSpec` class wires a single ORS environment into the three TRL trainer slots — `train_dataset`, `environment_factory`, `reward_funcs` — by exposing properties that map 1:1 to those kwarg names:

```python
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openreward import OpenRewardSpec

spec = OpenRewardSpec("Eigent/SETA", num_tasks=64)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=GRPOConfig(
        num_generations=2,
        max_steps=5,
        max_tool_calling_iterations=20,
        log_completions=True,
    ),
    train_dataset=spec.train_dataset,
    environment_factory=spec.environment_factory,
    reward_funcs=spec.reward_funcs,
)
trainer.train()
```

Under the hood `OpenRewardSpec` does three things, lazily on first access:

1. **`spec.train_dataset`**: derives a `datasets.Dataset` from the env's task list (one HTTP roundtrip via the SDK). Has at minimum `prompt`, `task_index`, plus per-task metadata columns folded in.
2. **`spec.environment_factory`**: returns a zero-arg callable that produces a fresh per-rollout adapter on each call. The adapter exposes one Python method per ORS tool, with a typed signature and docstring auto-generated from the env's JSON Schema. TRL's tool collector picks them up via `inspect.getmembers`.
3. **`spec.reward_funcs`**: an outcome-only reward function (last non-null reward in the trajectory) suitable for sparse-reward envs like SETA.

## Using a hub environment

Pass an [openreward.ai](https://openreward.ai) catalog name as the target. The SDK reads `OPENREWARD_API_KEY` from the environment for authentication.

```python
spec = OpenRewardSpec("Eigent/SETA", num_tasks=64)
```

## Using a self-hosted environment

Pass the URL directly. No API key is needed if your server doesn't enforce one.

```python
spec = OpenRewardSpec("https://my-org-my-env.hf.space", env_name="my_env")
```

> [!IMPORTANT]
> The `openreward` SDK by default expects a two-subdomain platform layout (`api.<host>` for stateless calls and `sessions.<host>` for SSE-based session calls). For **single-host** self-hosted servers (one URL serving everything), set the override env vars below before constructing `OpenRewardSpec`:
>
> ```python
> import os
>
> URL = "https://my-org-my-env.hf.space"
> os.environ["OPENREWARD_API_URL"]     = URL
> os.environ["OPENREWARD_SESSION_URL"] = URL
>
> spec = OpenRewardSpec(URL, env_name="my_env")
> ```

## Running a minimal environment locally

The fastest way to try the integration end-to-end without external dependencies is a tiny ORS server defined with the `openreward` SDK's `Environment` + `Server` scaffolding. The example below is a complete `echo` environment — the model wins by calling `echo(text=...)` with the task's target string.

```python
# server.py
from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool


class EchoTaskSpec(BaseModel):
    target: str

class EchoParams(BaseModel):
    text: str


class EchoEnvironment(Environment):
    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = EchoTaskSpec.model_validate(task_spec)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        return [{"target": "hello"}, {"target": "world"}]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(type="text", text=f"Echo '{self.config.target}' to win.")]

    @tool
    async def echo(self, params: EchoParams) -> ToolOutput:
        """Submit a string. Reward 1.0 + finished if it matches the target.

        Args:
            text: The string to echo back.
        """
        correct = params.text == self.config.target
        return ToolOutput(
            blocks=[TextBlock(type="text", text="match" if correct else "no match")],
            reward=1.0 if correct else 0.0,
            finished=correct,
        )


if __name__ == "__main__":
    Server([EchoEnvironment]).run(host="0.0.0.0", port=8000)
```

Run it:

```bash
pip install openreward fastapi uvicorn pydantic
python server.py     # listens on :8000
```

Then point `OpenRewardSpec` at it (with the URL overrides described above):

```python
import os
URL = "http://127.0.0.1:8000"
os.environ["OPENREWARD_API_URL"]     = URL
os.environ["OPENREWARD_SESSION_URL"] = URL

from trl.experimental.openreward import OpenRewardSpec
spec = OpenRewardSpec(URL, env_name="echoenvironment")
print(spec.train_dataset)        # 2 rows, task_index + target columns
```

This is also the fixture pattern used by TRL's own tests — see [`trl-internal-testing/openreward-echo-env`](https://huggingface.co/spaces/trl-internal-testing/openreward-echo-env) for the deployed Space.

## Selecting tasks

`OpenRewardSpec` accepts either a count or an explicit index list:

```python
spec = OpenRewardSpec("Eigent/SETA", num_tasks=10)                      # first 10 tasks
spec = OpenRewardSpec("Eigent/SETA", indices=[0, 5, 13, 27])            # specific indices
spec = OpenRewardSpec("Eigent/SETA", indices=list(range(50, 100)))      # range
```

`num_tasks` and `indices` are mutually exclusive and both fetch only the tasks they need (no full task list scan).

## How tool binding works

At construction the spec calls the env's `/tools` endpoint to fetch a list of tool specs (each with a name, description, and JSON Schema for arguments). For each tool it generates a Python method on the per-rollout adapter with a typed signature and a docstring derived from the schema. So `transformers.utils.get_json_schema` and TRL's `inspect.getmembers(env, ismethod)` both produce the right tool schema for the model with no per-env wrapper code.

If a tool description contains characters that aren't safe to splice into Python source, the binder falls back to a sanitized form so binding never fails on real envs.

## Reward functions

`spec.reward_funcs` defaults to an outcome-only reward — for each rollout it returns the last non-null reward observed during the trajectory. This is the right default for sparse-reward envs (e.g. SETA, where only `submit_solution` returns a non-null reward).

If you want a custom reward, write a regular TRL reward function and pass it directly:

```python
def my_reward(environments, **kwargs) -> list[float]:
    return [env.reward * 2.0 for env in environments]   # double the env reward, etc.

trainer = GRPOTrainer(
    ...,
    reward_funcs=my_reward,
)
```

The per-rollout adapter exposes the running state TRL needs — `env.reward`, `env.rewards`, `env.metadata`, `env.finished`, `env.last_output` — for arbitrary post-hoc reward shaping.

## OpenRewardSpec

[[autodoc]] trl.experimental.openreward.OpenRewardSpec

## Limitations

- The integration is in `trl.experimental` — APIs may change. Set `TRL_EXPERIMENTAL_SILENCE=1` to silence the warning in CI logs.
- Currently exposes a single `OpenRewardSpec` covering one environment; multi-environment training (à la the OpenEnv "meta-environment" pattern) is not supported yet.
- Long-running rollouts (>15 min per episode) need a keepalive ping — not yet wired.

## Reference

- [Open Reward Standard](https://openrewardstandard.io)
- [OpenReward platform](https://openreward.ai)
- [`openreward` Python SDK](https://pypi.org/project/openreward/)
- [Echo env Space — `trl-internal-testing/openreward-echo-env`](https://huggingface.co/spaces/trl-internal-testing/openreward-echo-env)
