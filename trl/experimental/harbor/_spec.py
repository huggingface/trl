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

"""User-facing spec for the Harbor × TRL integration (mirror of ``OpenRewardSpec``).

Construct **one** ``HarborSpec`` and read three properties off it — ``.train_dataset``, ``.environment_factory``,
``.reward_funcs`` — each plugging into the matching ``GRPOTrainer`` kwarg:

```python
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.harbor import HarborSpec

spec = HarborSpec("AdithyaSK/data_agent_rl_environment_train", agent="bash", num_tasks=64)

trainer = GRPOTrainer(
    model="Qwen/Qwen3.5-4B",
    args=GRPOConfig(num_generations=8, max_steps=50, max_tool_calling_iterations=25),
    train_dataset=spec.train_dataset,
    environment_factory=spec.environment_factory,
    reward_funcs=spec.reward_funcs,
)
trainer.train()
```

A Harbor *task* is a directory (``instruction.md`` + ``task.toml`` + ``environment/`` + ``tests/``); the dataset is a
tree of them. The ``environment_factory`` env runs Harbor in-process (see ``_env.py``), so ``harbor`` must be installed
in the same interpreter (``pip install trl[harbor]``, Python >= 3.12). The **base agent** (harness/tool surface) is
selected by ``agent=`` — ``"bash"`` today, or a custom ``HarborEnv`` subclass.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import cached_property, partial
from pathlib import Path
from typing import Any

from ._env import AGENTS, HarborEnv


def _outcome_reward_func(environments=None, environment_reward=None, **_) -> list[float]:
    """Default reward: the Harbor verifier's scalar per rollout.

    `GRPOTrainer` passes the live env instances as `environments=` (read `env.reward`); `AsyncGRPOTrainer` runs envs in
    its rollout worker and passes the already-captured per-rollout rewards as `environment_reward=`. Support both so
    the same spec plugs into either trainer.
    """
    if environment_reward is not None:
        return [float(r) for r in environment_reward]
    return [float(env.reward) for env in environments]


def _resolve_agent(agent: str | type[HarborEnv]) -> type[HarborEnv]:
    """Resolve the `agent=` selector to a `HarborEnv` subclass.

    Accepts a `HarborEnv` subclass, a built-in name (`"bash"`), a module import path (`"pkg.module:Class"`), or a file
    path (`"path/to/file.py:Class"`).
    """
    if isinstance(agent, type):
        cls = agent
    elif agent in AGENTS:
        cls = AGENTS[agent]
    elif ":" in agent:
        import importlib
        import importlib.util

        target, _, cls_name = agent.rpartition(":")  # rpartition: don't split a Windows drive (``D:\...``)
        if target.endswith(".py") or os.path.sep in target:  # file path -> load module from file
            spec = importlib.util.spec_from_file_location(Path(target).stem, target)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:  # dotted module path on sys.path
            module = importlib.import_module(target)
        cls = getattr(module, cls_name)
    else:
        raise ValueError(
            f"Unknown agent {agent!r}; use a built-in name ({sorted(AGENTS)}), an import path "
            "'pkg.module:Class', a file path 'path/to/file.py:Class', or a HarborEnv subclass."
        )
    if not (isinstance(cls, type) and issubclass(cls, HarborEnv)):
        raise TypeError(f"agent {agent!r} must resolve to a HarborEnv subclass, got {cls!r}")
    return cls


def _read_task_meta(task_dir: Path) -> dict[str, Any]:
    """Pull a few useful fields out of ``task.toml`` for the dataset rows / reward funcs."""
    try:
        import tomllib  # stdlib on Python 3.11+; lazy so the module imports on 3.10 (e.g. doc build)

        cfg = tomllib.loads((task_dir / "task.toml").read_text())
    except Exception:  # noqa: BLE001
        return {}
    meta = cfg.get("metadata", {})
    return {
        "gold_answer": meta.get("gold_answer"),
        "reward_mode": meta.get("reward_mode_initial"),
        "difficulty_level": meta.get("difficulty_level"),
        "kaggle_dataset": meta.get("kaggle_dataset_name"),
    }


class HarborSpec:
    """Single spec object that wires a Harbor task suite into a TRL trainer.

    Args:
        dataset (`str`):
            A Hugging Face dataset repo id holding a Harbor task tree (e.g.
            `"AdithyaSK/data_agent_rl_environment_train"`), or a local path to a directory containing a `tasks/`
            subtree. Each task is a dir with `instruction.md` / `task.toml` / `environment/` / `tests/`.
        agent (`str` or `type`, *optional*, defaults to `"bash"`):
            The base agent / harness — i.e. the tool surface the env exposes. One of: a built-in name (`"bash"`), an
            import path `"package.module:ClassName"`, a file path `"path/to/file.py:ClassName"`, or a
            [`~trl.experimental.harbor.HarborEnv`] subclass directly.
        environment_type (`str`, *optional*, defaults to `"docker"`):
            Harbor sandbox backend, passed through to Harbor (whatever it supports — `"docker"`, `"e2b"`, `"daytona"`,
            `"gke"`, `"modal"`, `"runloop"`, ...). Not validated here; Harbor validates. `"docker"` is Harbor's own
            default; pick `"e2b"` to offload sandboxing to the cloud.
        num_tasks (`int`, *optional*):
            Cap on the number of tasks pulled into the dataset. `None` uses every task in the tree.
        indices (`list[int]`, *optional*):
            Specific task indices (into the sorted task list). Mutually exclusive with `num_tasks`.
        include_metadata (`bool`, *optional*, defaults to `True`):
            Fold per-task `task.toml` metadata (gold_answer, difficulty, ...) into the dataset rows.
    """

    def __init__(
        self,
        dataset: str,
        *,
        agent: str | type[HarborEnv] = "bash",
        environment_type: str = "docker",
        num_tasks: int | None = None,
        indices: list[int] | None = None,
        include_metadata: bool = True,
    ) -> None:
        if num_tasks is not None and indices is not None:
            raise ValueError("Provide num_tasks or indices, not both.")
        self._dataset = dataset
        self._environment_type = environment_type
        self._num_tasks = num_tasks
        self._indices = indices
        self._include_metadata = include_metadata
        self._env_cls = _resolve_agent(agent)

    # ── public surface ──────────────────────────────────────────────

    @cached_property
    def _task_dirs(self) -> list[Path]:
        """Resolve the dataset to a sorted list of local task directories (downloading if needed)."""
        local = Path(self._dataset)
        if (local / "tasks").is_dir():
            root = local / "tasks"
        elif local.is_dir() and any(local.glob("*/task.toml")):
            root = local
        else:
            # Treat as an HF dataset repo id; download the task tree.
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(self._dataset, repo_type="dataset", allow_patterns=["tasks/**"]))
            root = path / "tasks"
        dirs = sorted(p.parent for p in root.glob("*/task.toml"))
        if self._indices is not None:
            dirs = [dirs[i] for i in self._indices]
        elif self._num_tasks is not None:
            dirs = dirs[: self._num_tasks]
        if not dirs:
            raise ValueError(f"No tasks (dir with task.toml) found under {root}")
        return dirs

    @cached_property
    def train_dataset(self):
        """A `datasets.Dataset` of tasks. Plugs into TRL's `train_dataset=`.

        Columns: `prompt` (empty user message — TRL appends the env's instruction from `reset`), `task_dir` (passed to
        `reset`), `task_index`, and per-task metadata when `include_metadata`.
        """
        from datasets import Dataset

        dirs = self._task_dirs
        rows: dict[str, list[Any]] = {
            "prompt": [[{"role": "user", "content": ""}] for _ in dirs],
            "task_dir": [str(d) for d in dirs],
            # task_index is the position in the sorted suite, so it matches the `indices` selector.
            "task_index": list(self._indices) if self._indices is not None else list(range(len(dirs))),
        }
        if self._include_metadata:
            metas = [_read_task_meta(d) for d in dirs]
            for key in ("gold_answer", "reward_mode", "difficulty_level", "kaggle_dataset"):
                rows[key] = [m.get(key) for m in metas]
        return Dataset.from_dict(rows)

    @cached_property
    def environment_factory(self) -> Callable[[], HarborEnv]:
        """Zero-arg callable returning a fresh harness env. Plugs into TRL's `environment_factory=`.

        Returns a `functools.partial` (not a closure) so it stays picklable — `AsyncGRPOTrainer` runs its rollout
        worker in a separate process and pickles the factory to it (closures/lambdas would fail).
        """
        return partial(self._env_cls, environment_type=self._environment_type)

    @property
    def reward_funcs(self) -> Callable[..., list[float]]:
        """Default outcome reward (Harbor verifier scalar). Plugs into TRL's `reward_funcs=`."""
        return _outcome_reward_func
