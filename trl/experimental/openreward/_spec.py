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

"""User-facing spec object for the OpenReward × TRL integration.

The user constructs **one** ``OpenRewardSpec`` (a thin specification holding the env target + a few options) and reads
three properties off of it — ``.train_dataset``, ``.environment_factory``, ``.reward_funcs`` — each of which plugs
directly into the matching ``GRPOTrainer`` kwarg:

```python
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openreward import OpenRewardSpec

spec = OpenRewardSpec("Eigent/SETA", num_tasks=64)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=GRPOConfig(num_generations=2, max_steps=5, max_tool_calling_iterations=20),
    train_dataset=spec.train_dataset,
    environment_factory=spec.environment_factory,
    reward_funcs=spec.reward_funcs,
)
trainer.train()
```

Backed by the official ``openreward`` SDK (lazy-imported); install with ``pip install trl[openreward]``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import cached_property
from typing import Any

from .environment import _import_openreward, _RolloutEnvironment


logger = logging.getLogger(__name__)


def _to_spec(task) -> dict[str, Any]:
    """Normalise an SDK ``Task`` (or already-a-dict) to its task_spec dict."""
    if isinstance(task, dict):
        return task
    return task.task_spec


def _outcome_only_reward_func(environments, **_):
    """Default reward function: last non-null reward in each rollout's trajectory.

    Suitable for sparse-outcome envs (e.g. SETA, where only `submit_solution` returns a non-null reward). Override by
    passing a different callable to ``reward_funcs=``.
    """
    return [env.reward for env in environments]


class OpenRewardSpec:
    """Single spec object that wires an ORS environment into a TRL trainer.

    Args:
        target (`str`):
            Either an openreward.ai catalog name (`"Eigent/SETA"`) or a URL pointing at any ORS server
            (`"https://you-seta.hf.space"`, `"http://localhost:8080"`). Auto-detected by the presence of `://` in the
            string.
        num_tasks (`int`, *optional*):
            Cap on the number of tasks pulled into the dataset. ``None`` uses every task the env exposes.
        split (`str`, *optional*, defaults to `"train"`):
            Which split's task list to draw from.
        indices (`list[int]`, *optional*):
            Specific task indices to train on. Mutually exclusive with ``num_tasks``. Useful for debugging or
            curriculum subsets.
        api_key (`str`, *optional*):
            ``OPENREWARD_API_KEY`` override. Only used when ``target`` is a catalog name.
        secrets (`dict[str, str]`, *optional*):
            Per-session secrets forwarded to ``env.session(secrets=)``.
        env_name (`str`, *optional*):
            Override for the env name to look up on the server. Rarely needed.
        include_metadata (`bool`, *optional*, defaults to `True`):
            Fold per-task metadata (`difficulty`, `category`, `tags`, ...) into the dataset rows so reward funcs can
            read them via TRL's ``inputs`` argument.
    """

    def __init__(
        self,
        target: str,
        *,
        num_tasks: int | None = None,
        split: str = "train",
        indices: list[int] | None = None,
        api_key: str | None = None,
        secrets: dict[str, str] | None = None,
        env_name: str | None = None,
        include_metadata: bool = True,
    ) -> None:
        if num_tasks is not None and indices is not None:
            raise ValueError("Provide num_tasks or indices, not both.")

        self._target = target
        self._is_url = "://" in target
        self._num_tasks = num_tasks
        self._split = split
        self._indices = indices
        self._api_key = api_key
        self._secrets = secrets
        self._env_name = env_name
        self._include_metadata = include_metadata

    # ── public surface ──────────────────────────────────────────────

    @cached_property
    def train_dataset(self):
        """A `datasets.Dataset` derived from the env's task list.

        Plugs directly into TRL's ``train_dataset=`` slot. Built lazily on first access. Has at minimum:
          - `prompt`: empty user message (TRL appends the env's prompt).
          - `task_index`: int passed to the adapter's `reset()`.
          - per-task metadata columns (when `include_metadata=True`).
        """
        from datasets import Dataset

        env = self._sdk_env

        # When the user asked for a specific subset (`num_tasks` cap or
        # explicit `indices`), fetch only those — `list_tasks` returns
        # the whole split (1376 entries for SETA, etc.) and is slow on
        # the platform. Per-index `get_task` is fast and cheap.
        if self._indices is not None:
            indexes = list(self._indices)
            task_specs = [_to_spec(env.get_task(self._split, i)) for i in indexes]
        elif self._num_tasks is not None:
            n_total = env.num_tasks(self._split)
            n = min(self._num_tasks, n_total)
            indexes = list(range(n))
            task_specs = [_to_spec(env.get_task(self._split, i)) for i in indexes]
        else:
            # No cap — fetch everything in one shot.
            try:
                tasks = env.list_tasks(self._split)
            except Exception:  # noqa: BLE001
                n = env.num_tasks(self._split)
                tasks = [env.get_task(self._split, i) for i in range(n)]
            task_specs = [_to_spec(t) for t in tasks]
            indexes = list(range(len(task_specs)))

        rows: dict[str, list[Any]] = {
            "prompt": [[{"role": "user", "content": ""}] for _ in indexes],
            "task_index": indexes,
        }

        if self._include_metadata and task_specs:
            metadata_keys: set[str] = set()
            for spec in task_specs:
                if isinstance(spec, dict):
                    metadata_keys.update(spec.keys())
            # Never let a task-spec key overwrite our reserved row columns
            # (e.g. an env that exposes a `prompt` task-spec field would
            # otherwise replace our chat-format prompt with a raw string).
            metadata_keys -= rows.keys()
            for key in sorted(metadata_keys):
                rows[key] = [s.get(key) if isinstance(s, dict) else None for s in task_specs]

        return Dataset.from_dict(rows)

    @cached_property
    def environment_factory(self) -> Callable[[], _RolloutEnvironment]:
        """Zero-arg callable that returns a fresh ``_RolloutEnvironment``.

        Plugs directly into TRL's ``environment_factory=`` slot. TRL calls this once per rollout at trainer
        construction time, so each rollout has an isolated ORS session. Reuses the spec's already-discovered SDK env +
        tool specs to skip per-env HTTP at trainer init.
        """
        # Pre-fetch tool specs once at the spec level.
        env = self._sdk_env
        tool_specs = env.list_tools()
        client = self._sdk_client

        # Each spec gets its own `_RolloutEnvironment` subclass so two specs
        # for different envs (different tool sets) never clobber each other's
        # bound methods on the shared parent class.
        rollout_cls = type(f"_RolloutEnvironment_{id(self):x}", (_RolloutEnvironment,), {})

        kwargs = {
            "split": self._split,
            "secrets": self._effective_secrets(),
            "env_name": self._env_name,
            "_client": client,
            "_env": env,
            "_tool_specs": tool_specs,
        }
        if self._is_url:
            kwargs["base_url"] = self._target
        else:
            kwargs["name"] = self._target
            if self._api_key:
                kwargs["api_key"] = self._api_key

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        def _make() -> _RolloutEnvironment:
            return rollout_cls(**kwargs)

        return _make

    @property
    def reward_funcs(self) -> Callable[..., list[float]]:
        """Default outcome-only reward function (last non-null reward per rollout).

        Plugs directly into TRL's ``reward_funcs=`` slot. Stable identity — module-level function, picklable for
        multi-process workers.
        """
        return _outcome_only_reward_func

    # ── internals ───────────────────────────────────────────────────

    def _effective_secrets(self) -> dict[str, str] | None:
        """Auto-forward ``OPENREWARD_API_KEY`` as the conventional ``api_key``
        per-session secret on platform mode (most envs need it)."""
        if self._secrets is not None or self._is_url:
            return self._secrets
        key = self._api_key or os.environ.get("OPENREWARD_API_KEY")
        return {"api_key": key} if key else None

    @cached_property
    def _sdk_client(self):
        """The shared ``openreward.OpenReward`` client used by `dataset` and `factory`."""
        openreward = _import_openreward()
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        elif not self._is_url and "OPENREWARD_API_KEY" in os.environ:
            kwargs["api_key"] = os.environ["OPENREWARD_API_KEY"]
        if self._is_url:
            kwargs["base_url"] = self._target
        return openreward.OpenReward(**kwargs)

    @cached_property
    def _sdk_env(self):
        """The shared SDK ``Environment`` handle."""
        if self._is_url:
            # Self-hosted single-env URL — pass any name; the SDK redirects.
            # Fallback to "env" matches `_RolloutEnvironment.__init__`.
            target = self._env_name or "env"
        else:
            target = self._target
        return self._sdk_client.environments.get(target)
