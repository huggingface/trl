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
"""The General domain: 925 enterprise knowledge-work tasks, each a workspace of office documents plus a few
business systems served as MCP tools, graded by a rubric whose items an LLM judges against a hidden answer key.

Every rollout gets one sandbox started from the task's image. Upstream runs a two-container pod, the agent in one
and the MCP servers plus the verifier in a sidecar it cannot see; here both halves share the sandbox, and the
anti-hack guard's path rules stand in for that wall (the verifier materials only reach the sandbox after the agent
stops)."""

from __future__ import annotations

import json
import os
import random
import re
import shlex
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import Sandbox, hf_hub_download, snapshot_download
from mimo_sandbox import DATASET, HFSandboxEnvironment, ProbeAgent, docker_image


os.environ.setdefault("MIMOAGENT_SILENT_STARTUP", "1")
from mimoagent.agents.antihack import AntiHackConfig  # noqa: E402
from mimoagent.tools.base import BaseTool, ToolException, ToolOutput  # noqa: E402


VENV_PYTHON = "/opt/openai-agents-venv/bin/python"
# `mcp_bridge_fast.py` next to this file, uploaded per sandbox, in place of the task's own bridge. Both speak the
# same one-shot protocol, but the task's imports the `mcp` package, whose import alone costs about twelve seconds
# inside a sandbox and is paid once per tool call; the replacement uses only the standard library.
MCP_BRIDGE = "/work/_setup/mcp_bridge_fast.py"
MCP_BRIDGE_SOURCE = Path(__file__).with_name("mcp_bridge_fast.py")
ONESHOT_MARKER = "__MCP_ONESHOT__"

# `config/agent/general/s3k.yaml` of XiaomiMiMo/verl: the agent the dataset was trained with.
SYSTEM_TEMPLATE = (
    "You are an agent, your current working directory is {{cwd}}.\n\n"
    "You can use the tools available to you to interact with the computer to assist the user in completing tasks.\n"
)
INSTANCE_TEMPLATE = "{{task}}\n"
# `TRAJECTORY_TIMEOUT` of `scripts/general/train.sh` upstream.
AGENT_TIMEOUT = 1200
TOOLS = [
    {"tool": "Bash", "config": {"timeout": 60}},
    {"tool": "Read"},
    {"tool": "Write"},
    {"tool": "Edit"},
    {"tool": "Grep"},
    {"tool": "Glob"},
]
# What the agent cannot reach upstream because it lives in the sidecar: the MCP servers' source and state, the
# verifier and the bridge machinery. Any tool call naming one of these gets the guard's dummy observation.
SIDECAR_PATHS = r"/work/(system|tools|_setup)|/installed-agent|/opt/openai-agents-venv|/logs/verifier"

# Appended to the task's own `verify.py` before it is uploaded. Upstream judges the rubric one item at a time, and
# the verifier runs inside the rollout's sandbox while the rollout still holds its slot, so a task's median of five
# LLM-judged items serializes five round trips into the critical path. Redefining the function at the end of the
# module is enough: `_grade_sc` resolves it as a module global when it calls it. Evidence extraction stays
# sequential, being CPU-bound and already cached per set of files.
PARALLEL_JUDGE = '''

def _run_llm_sc(items, ws, votes=1):
    """Judge the rubric items concurrently (appended by TRL's async_grpo_mimo_rl_oss example)."""
    import statistics
    from concurrent.futures import ThreadPoolExecutor

    if not items:
        return {}
    ev_cache = {}
    for it in items:
        key = tuple(sorted(it.get("files") or []))
        if key not in ev_cache:
            ev_cache[key] = _evidence_sc(ws, list(key))

    def _one(it):
        key = tuple(sorted(it.get("files") or []))
        item_txt = ("- [" + it["id"] + "] " + str(it.get("question", it.get("desc", "")))
                    + "(满足=1:" + str(it.get("pass_anchor", "")) + ")")
        prompt = _LLM_PROMPT.replace("{evidence}", ev_cache[key]).replace("{items}", item_txt)
        vals = []
        for _ in range(max(1, votes)):
            j = None
            for _p in range(2):
                j = _extract_json_sc(_chat_judge_sc(prompt))
                if j is not None:
                    break
            sc = (((j or {}).get("results") or {}).get(it["id"]) or {}).get("score")
            if isinstance(sc, (int, float)):
                vals.append(max(0.0, min(1.0, float(sc))))
        if not vals:
            raise _JudgeUnavailable("no usable verdict for item " + str(it.get("id")))
        return it["id"], {"score": round(statistics.median(vals), 4), "detail": "llm " + str(len(vals)) + "票"}

    with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
        return dict(pool.map(_one, items))
'''

# ============================================================================================================
# Tasks
# ============================================================================================================


def load_tasks(
    n_tasks: int,
    seed: int,
    categories: set[str] | None = None,
    language: str | None = None,
    instance_ids: list[str] | None = None,
) -> list[dict]:
    """The General split's `general_agent` rows, filtered, shuffled and cut to `n_tasks`, each with its task
    directory downloaded and its manifest parsed. The instance id encodes the category, the language and a tier:
    `s3k_0000_accounting_audit_tax_en_t1_rl_008`. `instance_ids` restricts the pool, e.g. to tasks the base policy
    sometimes solves."""
    rows = pd.read_parquet(hf_hub_download(DATASET, "general/train.parquet", repo_type="dataset"))
    tasks = []
    for extra in rows["extra_info"]:
        instance = json.loads(extra["instance_json"])
        if instance["dataset_type"] != "general_agent":
            continue
        match = re.fullmatch(r"s3k_\d+_(.+)_(en|zh)_t(\d)_rl_\d+", instance["instance_id"])
        instance |= {"category": match[1], "language": match[2], "tier": int(match[3])}
        if (categories and instance["category"] not in categories) or (language and instance["language"] != language):
            continue
        if instance_ids is not None and instance["instance_id"] not in instance_ids:
            continue
        tasks.append(instance)
    random.Random(seed).shuffle(tasks)
    tasks = tasks[:n_tasks]
    root = snapshot_download(
        DATASET, repo_type="dataset", allow_patterns=[f"general/{task['env_task_dir']}/*" for task in tasks]
    )
    for task in tasks:
        task["task_dir"] = Path(root) / "general" / task["env_task_dir"]
        task["manifest"] = json.loads((task["task_dir"] / "manifest.json").read_text())
    return tasks


# ============================================================================================================
# The task's business systems, as MCP tools
# ============================================================================================================


def bridge_command(url: str, server: str, *args: str) -> str:
    return shlex.join(["python3", MCP_BRIDGE, "--url", url, "--name", server, *args])


def bridge_result(result: dict[str, Any]) -> dict:
    """The bridge prints its one-shot result as a single marker-prefixed JSON line, so it survives whatever else
    lands on the merged stdout/stderr stream."""
    lines = [line for line in result["output"].splitlines() if ONESHOT_MARKER in line]
    if not lines:
        raise ToolException(f"MCP bridge produced no result; output tail:\n{result['output'][-800:]}")
    return json.loads(lines[-1].split(ONESHOT_MARKER, 1)[1])


class McpTool(BaseTool):
    """One tool of one of the task's MCP servers, called one-shot through the in-sandbox bridge, as verl's
    `recipes/general/mcp_proxy.py` does. The model sees `mcp__<server>__<function>` in its tool list."""

    def __init__(self, server: str, url: str, spec: dict):
        super().__init__({})
        self.server = server
        self.url = url
        self.spec = spec

    @property
    def name(self) -> str:
        return f"mcp__{self.server}__{self.spec['name']}"

    @property
    def description(self) -> str:
        return self.spec.get("description") or f"MCP tool {self.spec['name']} on server {self.server}."

    def get_function_parameters(self) -> dict[str, Any]:
        return self.spec.get("inputSchema") or {"type": "object", "properties": {}}

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        arguments = json.dumps(params if isinstance(params, dict) else {}, ensure_ascii=False)
        command = bridge_command(self.url, self.server, "--call", self.spec["name"], "--args-json", arguments)
        payload = bridge_result(context["env"].execute(command, timeout=120))
        if not payload.get("ok"):
            return ToolOutput(output=f"[mcp error] {payload.get('error', 'unknown')}", success=False)
        text = payload.get("content", "")
        if not text and payload.get("structured") is not None:
            text = json.dumps(payload["structured"], ensure_ascii=False)
        return ToolOutput(output=text, success=not payload.get("is_error", False))


def discover_mcp_tools(env: HFSandboxEnvironment, servers: list[dict]) -> list[McpTool]:
    tools = []
    for server in servers:
        payload = bridge_result(env.execute(bridge_command(server["url"], server["name"], "--list"), timeout=120))
        if not payload.get("ok"):
            raise RuntimeError(f"{server['name']}: {payload.get('error', 'tools/list failed')}")
        tools += [McpTool(server["name"], server["url"], spec) for spec in payload["tools"]]
    return tools


# ============================================================================================================
# Session and factory
# ============================================================================================================


class GeneralTaskSession:
    """One rollout: a sandbox holding the task's workspace and MCP servers, and the agent that works in it. `run`
    drives the agent to its end, `verify` grades the sandbox with the task's own verifier."""

    def __init__(
        self,
        *,
        task: dict,
        sandbox: Sandbox,
        env: HFSandboxEnvironment,
        agent: ProbeAgent,
        judge_env: dict,
        verify_timeout: int,
    ):
        self.task = task
        self.sandbox = sandbox
        self.env = env
        self.agent = agent
        self.judge_env = judge_env
        self.verify_timeout = verify_timeout

    def run(self) -> tuple[str, str]:
        return self.agent.run(task=self.task["problem_statement"])

    def grade(self, final_message: str) -> dict:
        # Upstream persists the agent's final reply as `workspace/answer.md` unless the agent wrote one itself; the
        # rubric items read that file. The verifier materials, answer keys included, reach the sandbox only now.
        if final_message.strip() and self.env.execute("test -s /work/workspace/answer.md")["returncode"] != 0:
            self.sandbox.files.write("/work/workspace/answer.md", final_message)
        verifier = self.task["manifest"]["verifier"]
        for upload in verifier["uploads"]:
            source = self.task["task_dir"] / upload["source"]
            if not source.exists():
                continue
            if source.name == "verify.py":
                self.sandbox.files.write(upload["target"], source.read_text() + PARALLEL_JUDGE)
            else:
                self.sandbox.files.upload(str(source), upload["target"])
        reward_file, detail_file = verifier["reward_file"], verifier["reward_detail_file"]
        self.env.execute(
            f"mkdir -p {os.path.dirname(reward_file)} /tmp/agent_output && rm -f {reward_file} {detail_file}"
        )
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.judge_env.items())
        result = self.env.execute(f"{env_prefix} {verifier['command']}", cwd="/work", timeout=self.verify_timeout)
        reward = (
            json.loads(self.sandbox.files.read_text(reward_file)) if self.sandbox.files.exists(reward_file) else {}
        )
        detail = (
            json.loads(self.sandbox.files.read_text(detail_file)) if self.sandbox.files.exists(detail_file) else {}
        )
        return {
            "reward": reward.get("reward"),
            "reward_error": reward.get("reward_error") or ("verifier_timed_out" if result["reason"] != "ok" else None),
            "judge_error": (detail.get("detail") or {}).get("judge_error"),
            "items": {item["id"]: item["score"] for item in detail.get("results", [])},
            "verifier_output": result["output"][-2000:],
        }

    def close(self) -> None:
        self.sandbox.kill()


def judge_env(url: str, key: str, model: str) -> dict[str, str]:
    """The verifier's judge settings, as the environment it reads them from. It falls back along a chain when a
    judge fails (a rate limit, an empty reply), so two further router models back the primary one; `GA_JUDGE_ROTATE=0`
    keeps the primary first for every rollout instead of spreading load across the chain. The verifier caps a verdict
    at 4000 tokens and cannot turn thinking off, so a reasoning model as primary loses the verdicts whose reasoning
    overruns the cap (13% with GLM-5.3); the default is a non-reasoning model."""
    fallbacks = [m for m in ("zai-org/GLM-5.3", "zai-org/GLM-5.2") if m != model]
    env = {"GA_JUDGE_URL": url, "GA_JUDGE_KEY": key, "GA_JUDGE_MODEL": model, "GA_JUDGE_API": "chat"}
    for i, fallback in enumerate(fallbacks, start=2):
        env |= {
            f"GA_JUDGE_URL{i}": url,
            f"GA_JUDGE_KEY{i}": key,
            f"GA_JUDGE_MODEL{i}": fallback,
            f"GA_JUDGE_API{i}": "chat",
        }
    return env | {"GA_JUDGE_ROTATE": "0", "VERIFY_DETERMINISTIC": "1", "VERIFY_AGENT_JUDGE": "1"}


class GeneralTaskSessionFactory:
    def __init__(
        self,
        *,
        make_model: Callable[[], Any],
        max_observation_length: int,
        step_limit: int,
        agent_timeout: int,
        verify_timeout: int,
        judge_env: dict[str, str],
        flavor: str,
        transcripts_dir: Path,
    ):
        self.make_model = make_model
        self.max_observation_length = max_observation_length
        self.step_limit = step_limit
        self.agent_timeout = agent_timeout
        self.verify_timeout = verify_timeout
        self.judge_env = judge_env
        self.flavor = flavor
        self.transcripts_dir = transcripts_dir

    def create(self, task: dict, sample: int) -> GeneralTaskSession:
        manifest = task["manifest"]
        # A dedicated sandbox, not a pooled one: a pooled sandbox is an unprivileged home on a shared host, so it
        # cannot write the `/work`, `/installed-agent` and `/logs` paths a task installs itself into.
        sandbox = Sandbox.create(
            image=docker_image(task["docker_image"]),
            flavor=self.flavor,
            idle_timeout=self.agent_timeout + self.verify_timeout,
            start_timeout=600,
        )
        try:
            env = HFSandboxEnvironment(sandbox, cwd=manifest["cwd"], timeout=1800)
            self._install(env, task)
            antihack = AntiHackConfig(enabled=True)
            antihack.bash_patterns.append(SIDECAR_PATHS)
            antihack.path_patterns.append(SIDECAR_PATHS)
            agent = ProbeAgent(
                self.make_model(),
                env,
                deadline=time.monotonic() + self.agent_timeout,
                system_template=SYSTEM_TEMPLATE,
                instance_template=INSTANCE_TEMPLATE,
                tools=TOOLS,
                step_limit=self.step_limit,
                max_observation_length=self.max_observation_length,
                antihack=antihack,
                msg_path=self.transcripts_dir / f"{task['instance_id']}-{sample}.log",
            )
            for tool in discover_mcp_tools(env, manifest["mcp_servers"]):
                agent.tool_registry.register(tool)
            agent._tool_definitions = agent.tool_registry.get_function_definitions()
        except Exception:
            sandbox.kill()
            raise
        return GeneralTaskSession(
            task=task,
            sandbox=sandbox,
            env=env,
            agent=agent,
            judge_env=self.judge_env,
            verify_timeout=self.verify_timeout,
        )

    def _install(self, env: HFSandboxEnvironment, task: dict) -> None:
        """Lay the task out as its manifest says, both containers' uploads into the one sandbox, and start the
        MCP servers. The verifier's uploads are not part of this: they carry the answer keys."""
        manifest = task["manifest"]
        result = env.execute(
            f"python3 -m venv {os.path.dirname(os.path.dirname(VENV_PYTHON))} && {VENV_PYTHON} -m pip install -q 'mcp<2'",
            timeout=300,
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"MCP venv install failed:\n{result['output'][-2000:]}")
        with tempfile.NamedTemporaryFile(suffix=".tar") as f:
            with tarfile.open(f.name, "w", dereference=True) as tar:
                for upload in manifest["uploads"]:
                    tar.add(task["task_dir"] / upload["source"], arcname=upload["target"].lstrip("/"))
            env.sandbox.files.upload(f.name, "/tmp/task.tar")
        env.execute("tar -xf /tmp/task.tar --no-same-owner -C / && rm /tmp/task.tar")
        env.sandbox.files.write(MCP_BRIDGE, MCP_BRIDGE_SOURCE.read_text())
        setup = manifest["setup"]
        result = env.execute(setup["command"], timeout=setup.get("timeout_sec") or 300)
        if result["returncode"] != 0:
            raise RuntimeError(f"MCP setup failed:\n{result['output'][-2000:]}")
