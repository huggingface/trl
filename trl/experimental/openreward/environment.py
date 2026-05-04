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

"""TRL `environment_factory` adapter for any ORS-compliant server.

Wraps the official ``openreward`` SDK so platform quirks (X-Secrets encoding, ephemeral session for discovery,
sticky-routing subdomain, SSE chunk reassembly, ping keepalive, …) are handled by code that ships with the spec. Our
job is only to expose the SDK's per-rollout session as a TRL-compatible class with dynamically-bound tool methods.

Install: ``pip install trl[openreward]``.
"""

from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


def _import_openreward():
    """Lazy-import the openreward package with a friendly error message."""
    try:
        import openreward

        return openreward
    except ImportError as e:
        raise ImportError(
            "trl.experimental.openreward requires the `openreward` package. "
            "Install with `pip install trl[openreward]`."
        ) from e


# ── JSON Schema → Python type mapping (used by dynamic tool binding) ──

_JSON_TYPE_TO_PY: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

_PY_TYPE_OBJECTS: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}

_PY_TYPE_FROM_VALUE: dict[type, str] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    list: "list",
    dict: "dict",
}


def _resolve_param_type(pdef: dict[str, Any]) -> str:
    """Pick the best Python type name for one JSON-Schema property.

    Handles primitive ``"type"``, Pydantic's ``anyOf``/``oneOf`` form for ``Optional[T]``, and falls back to the type
    of ``default``. Critical because transformers' tool-schema generator uses our Python annotations to build the
    schema the model sees.
    """
    t = pdef.get("type")
    if isinstance(t, str) and t in _JSON_TYPE_TO_PY:
        return _JSON_TYPE_TO_PY[t]
    for key in ("anyOf", "oneOf"):
        for opt in pdef.get(key) or []:
            if isinstance(opt, dict):
                inner = opt.get("type")
                if isinstance(inner, str) and inner in _JSON_TYPE_TO_PY and inner != "null":
                    return _JSON_TYPE_TO_PY[inner]
    if "default" in pdef and pdef["default"] is not None:
        for value_type, name in _PY_TYPE_FROM_VALUE.items():
            if isinstance(pdef["default"], value_type):
                return name
    return "str"


# ────────────────────────────────────────────────────────────────────


class _RolloutEnvironment:
    """Per-rollout TRL adapter backed by the ``openreward`` SDK.

    **Internal class.** Users don't construct this directly; they create an :class:`OpenRewardSpec` and pass
    ``spec.environment_factory`` to ``GRPOTrainer``. TRL's trainer then constructs one instance of this per rollout
    slot.

    The tool surface is built once at construction by reading the env's tools and binding one Python method per ORS
    tool — so swapping the URL or env name is the only thing needed to train against a different environment.

    Args:
        name (`str`, *optional*):
            openreward.ai catalog name (e.g. ``"Eigent/SETA"``). Mutually exclusive with `base_url`. Requires
            ``OPENREWARD_API_KEY`` (env var or `api_key=`).
        base_url (`str`, *optional*):
            Direct URL of an ORS server (HF Space, local Docker, etc.). Mutually exclusive with `name`.
        env_name (`str`, *optional*):
            Name to look up on the server. Defaults to the canonical name parsed from `name`/`base_url`.
        split (`str`, *optional*, defaults to `"train"`):
            Split passed to ``env.session(split=, index=)``.
        api_key (`str`, *optional*):
            Override for ``OPENREWARD_API_KEY``. Only used with `name=`.
        secrets (`dict[str, str]`, *optional*):
            Per-session secrets. Forwarded to ``env.session(secrets=)``; the SDK encodes them and adds platform-domain
            entries.
        timeout (`float`, *optional*):
            Reserved for future use. The SDK manages its own timeouts.

    Attributes:
        reward (`float`):
            Last non-null reward in the trajectory (outcome-only convention).
        rewards (`list[float | None]`):
            Per-step reward sequence in tool-call order.
        metadata (`list[dict | None]`):
            Per-step ``ToolOutput.metadata`` dicts.
        finished (`bool`):
            True after a tool returned ``finished: true``.
        last_output (`str`):
            Joined text of the most recent tool result.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        base_url: str | None = None,
        env_name: str | None = None,
        split: str = "train",
        api_key: str | None = None,
        secrets: dict[str, str] | None = None,
        timeout: float | None = None,
        _client: Any = None,
        _env: Any = None,
        _tool_specs: list[Any] | None = None,
    ) -> None:
        if (name is None) == (base_url is None):
            raise ValueError("Provide exactly one of: name, base_url.")

        # Reuse a pre-built client/env when the spec has already discovered
        # them (saves N HTTP calls at trainer init); otherwise build fresh.
        if _client is None:
            openreward = _import_openreward()
            client_kwargs: dict[str, Any] = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            elif name and "OPENREWARD_API_KEY" in os.environ:
                client_kwargs["api_key"] = os.environ["OPENREWARD_API_KEY"]
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = openreward.OpenReward(**client_kwargs)
        else:
            self._client = _client

        if _env is None:
            target = name if name is not None else env_name or ""
            if base_url and not target:
                # Self-hosted single-env URL — pass any name; SDK redirects.
                target = "env"
            self._env = self._client.environments.get(target)
        else:
            self._env = _env

        self._split = split
        self._secrets = secrets
        self._session = None  # entered on reset()

        # Episode state — read by the trainer's reward_func.
        self.reward: float = 0.0
        self.rewards: list[float | None] = []
        self.metadata: list[dict[str, Any] | None] = []
        self.finished: bool = False
        self.last_output: str = ""

        # Bind one Python method per server-side tool. Pre-fetched specs
        # come from the spec object; otherwise we ask the SDK ourselves.
        specs = _tool_specs if _tool_specs is not None else self._env.list_tools()
        for spec in specs:
            _bind_tool_method(self.__class__, _spec_to_dict(spec))

    # ── TRL contract ─────────────────────────────────────────────────

    def reset(
        self,
        *,
        task_index: int = 0,
        task_spec: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        """Open a fresh ORS session for this rollout.

        Closes any prior session, starts a new one via the SDK's ``env.session(...)`` context manager, fetches the
        prompt, and returns its text (which TRL appends to the user message).
        """
        self._teardown_session()
        self.reward = 0.0
        self.rewards = []
        self.metadata = []
        self.finished = False
        self.last_output = ""

        if task_spec is not None:
            from openreward.api.environments.client import Task

            task = Task(
                server_name=self._env.server if hasattr(self._env, "server") else self._env.name,
                environment_name=self._env.name,
                task_spec=task_spec,
                namespace=self._env.namespace,
            )
            cm = self._env.session(task=task, secrets=self._secrets)
        else:
            cm = self._env.session(split=self._split, index=int(task_index), secrets=self._secrets)

        self._session = cm.__enter__()
        prompt = self._session.get_prompt()
        return _join_text_blocks(prompt)

    # Leading underscore so TRL's tool collector at grpo_trainer.py:502-506
    # excludes this from the model's tool surface.
    def _close(self) -> None:
        """Tear down the active session (best-effort)."""
        self._teardown_session()

    # ── helpers ──────────────────────────────────────────────────────

    def _teardown_session(self) -> None:
        if self._session is None:
            return
        try:
            # The SDK's Session context-manager exit deletes both the env
            # state and the sid. We look up the original CM from the SDK.
            self._session.__exit__(None, None, None) if hasattr(self._session, "__exit__") else None
        except Exception as e:  # noqa: BLE001
            logger.debug("session teardown failed: %s", e)
        self._session = None

    def _call_ors_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Invoke a server-side tool and update episode state."""
        if self._session is None:
            raise RuntimeError("Cannot call a tool before reset() opens a session.")

        try:
            out = self._session.call_tool(tool_name, tool_input)
        except Exception as e:  # noqa: BLE001
            self.rewards.append(None)
            self.metadata.append(None)
            self.last_output = f"Error: {e}"
            return self.last_output

        text = _join_text_blocks(getattr(out, "blocks", []) or []) or "(no output)"

        raw_reward = getattr(out, "reward", None)
        step_reward: float | None = None
        if raw_reward is not None:
            try:
                step_reward = float(raw_reward)
            except (TypeError, ValueError):
                step_reward = None
        self.rewards.append(step_reward)
        if step_reward is not None:
            self.reward = step_reward  # last-non-null wins (outcome-only convention)

        md = getattr(out, "metadata", None)
        self.metadata.append(md if isinstance(md, dict) else None)

        self.finished = bool(getattr(out, "finished", False))
        self.last_output = text
        return text


# ── dynamic tool binding ─────────────────────────────────────────────


def _bind_tool_method(cls: type, spec: dict[str, Any]) -> None:
    """Generate a typed Python method for one ORS tool spec.

    Signature comes from the JSON Schema; docstring is Google-style so ``transformers.utils.get_json_schema`` (used by
    vLLM) can produce a correct tool schema for the model.
    """
    tool_name = spec["name"]
    if hasattr(cls, tool_name) and callable(getattr(cls, tool_name)):
        # Already bound by an earlier instance — idempotent skip.
        return

    description = spec.get("description") or f"Call the {tool_name} tool."
    schema = spec.get("input_schema") or {}
    properties: dict[str, dict[str, Any]] = schema.get("properties") or {}
    required: set[str] = set(schema.get("required") or [])

    # Required params must come before optional ones in Python — sort.
    required_props = [(n, p) for n, p in properties.items() if n in required]
    optional_props = [(n, p) for n, p in properties.items() if n not in required]

    params_src: list[str] = []
    annotations: dict[str, type] = {}
    args_doc_lines: list[str] = []
    for pname, pdef in required_props + optional_props:
        py_type = _resolve_param_type(pdef)
        annotations[pname] = _PY_TYPE_OBJECTS[py_type]
        if pname in required:
            params_src.append(f"{pname}: {py_type}")
        else:
            default = pdef.get("default", None)
            params_src.append(f"{pname}: {py_type} = {default!r}")
        pdesc = pdef.get("description") or pdef.get("title") or pname
        args_doc_lines.append(f"        {pname}: {pdesc}")

    args_section = ("\n\n    Args:\n" + "\n".join(args_doc_lines)) if args_doc_lines else ""
    src = (
        f"def {tool_name}(self, {', '.join(params_src)}) -> str:\n"
        f'    """{description}{args_section}"""\n'
        f"    _kwargs = dict(locals()); _kwargs.pop('self', None)\n"
        f"    return self._call_ors_tool({tool_name!r}, _kwargs)\n"
    )
    ns: dict[str, Any] = {}
    exec(src, ns)
    fn = ns[tool_name]
    fn.__qualname__ = f"{cls.__name__}.{tool_name}"
    fn.__annotations__ = {**annotations, "return": str}
    setattr(cls, tool_name, fn)


# ── small utilities ──────────────────────────────────────────────────


def _spec_to_dict(spec: Any) -> dict[str, Any]:
    """Normalise SDK ``ToolSpec`` and dict shapes to the same dict form."""
    if isinstance(spec, dict):
        return spec
    return {
        "name": spec.name,
        "description": getattr(spec, "description", None),
        "input_schema": getattr(spec, "input_schema", None) or {},
    }


def _join_text_blocks(blocks: list[Any]) -> str:
    """Concatenate the ``text`` field of every text block in order.

    Accepts both SDK ``TextBlock`` objects and plain dicts.
    """
    if not blocks:
        return ""
    parts: list[str] = []
    for b in blocks:
        text = getattr(b, "text", None) if not isinstance(b, dict) else b.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)
