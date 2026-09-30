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

# /// script
# dependencies = [
#     "mimoagent @ git+https://github.com/XiaomiMiMo/MiMo-Agent.git",
#     "huggingface_hub>=1.31",
#     "pandas",
#     "pyarrow",
# ]
# ///

"""Probe a policy on the General domain of `XiaomiMiMo/MiMo-V2.6-RL-oss`, no training.

The General domain is 925 knowledge-work tasks (accounting, legal, HR, healthcare ops, ...). Each task is a
directory: a `workspace/` of office documents the agent reads, a few business systems served as MCP tools over
SQLite state, and a verifier that grades the agent's final reply against rubric items, most of them judged by an
LLM against a hidden answer key. The agent is MiMo's own CC-style agent (`mimoagent`, Bash/Read/Write/Edit/Grep/Glob
plus the task's MCP tools), driven with the same prompts, tool config and anti-hack guard as
`XiaomiMiMo/verl`'s `recipes/general`.

Every rollout gets one Hugging Face sandbox started from the task's image. Upstream runs a two-container pod, the
agent in one and the MCP servers plus the verifier in a sidecar it cannot see; here both halves share the sandbox,
and the anti-hack guard's path rules stand in for that wall (the verifier materials only reach the sandbox after the
agent stops). The LLM judge is any OpenAI-compatible chat endpoint reachable from the sandbox, by default Qwen3-235B on the
Hugging Face router.

Prints the reward distribution, the pass rate, the mean per task and how many tasks have a reward spread across
their samples, which is the signal GRPO trains on. Writes one JSON line per rollout and one transcript per rollout.

```sh
vllm serve Qwen/Qwen3-8B --max-model-len 40960 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --generation-config vllm

HF_TOKEN=... python examples/async_grpo_mimo_rl_oss/probe_general.py --model Qwen/Qwen3-8B \
    --n-tasks 32 --samples-per-task 4 --output probe.jsonl

# no GPU: the policy served by the router too
python examples/async_grpo_mimo_rl_oss/probe_general.py --model Qwen/Qwen3-8B \
    --base-url https://router.huggingface.co/v1 --api-key $HF_TOKEN --n-tasks 2 --samples-per-task 1
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shlex
import statistics
import tarfile
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import Sandbox, hf_hub_download, snapshot_download


os.environ.setdefault("MIMOAGENT_SILENT_STARTUP", "1")
from mimoagent.agents.antihack import AntiHackConfig  # noqa: E402
from mimoagent.agents.base import LimitsExceeded  # noqa: E402
from mimoagent.agents.cc import CCAgent  # noqa: E402
from mimoagent.environments import TransportError  # noqa: E402
from mimoagent.models.openai_chat import OpenAIChatModel  # noqa: E402
from mimoagent.tools.base import BaseTool, ToolException, ToolOutput  # noqa: E402


DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DOCKER_REPOSITORY = "docker.io/xiaomimimo/mimo-v2.6-rl-oss"
# The sidecar entrypoint hardcodes this interpreter for the MCP servers, and the public image ships MCP 2.x where
# the task's server and bridge scripts are written for 1.x. Each sandbox gets the venv the scripts expect.
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
# mimoagent environment and MCP tools over a Hugging Face sandbox
# ============================================================================================================


class HFSandboxEnvironment:
    """mimoagent `Environment` running each command in a Hugging Face sandbox. Same contract as its
    `KubernetesEnvironment`: a fresh `bash -c` per command, stderr merged into stdout, a `reason` the Bash tool
    reads, and files shipped in by `copy_to`."""

    def __init__(self, sandbox: Sandbox, *, cwd: str, timeout: int):
        self.sandbox = sandbox
        self.config = {"cwd": cwd, "timeout": timeout}

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> dict[str, Any]:
        # Output goes through a file read back as bytes rather than through the command's stream: the sandbox client
        # splits that stream on U+0085, U+2028 and U+2029 as well as on newlines, so a file holding one of them
        # breaks the whole command. A file also lets a command leave daemons behind, as the MCP setup does.
        output_file = f"/tmp/out-{uuid.uuid4().hex}"
        try:
            result = self.sandbox.run(
                ["bash", "-c", f"exec >{output_file} 2>&1\n" + command],
                cwd=cwd or self.config["cwd"],
                timeout=timeout or self.config["timeout"],
                check=False,
            )
            output = self.sandbox.files.read(output_file).decode("utf-8", errors="replace")
        except Exception as e:
            raise TransportError(str(e)) from e
        if result.timed_out:
            return {"output": output, "returncode": 124, "reason": "pod_timeout"}
        return {"output": output, "returncode": result.exit_code, "reason": "ok"}

    def execute_detached(self, command: str, cwd: str = "", timeout: int | None = None, **_) -> dict[str, Any]:
        return self.execute(command, cwd=cwd, timeout=timeout)

    def copy_to(self, src_path: str, dest_path: str, **_) -> None:
        if not os.path.isdir(src_path):
            self.execute(f"mkdir -p {shlex.quote(os.path.dirname(dest_path))}")
            self.sandbox.files.upload(src_path, dest_path)
            return
        archive = f"/tmp/copy_to-{uuid.uuid4().hex}.tar"
        with tempfile.NamedTemporaryFile(suffix=".tar") as f:
            with tarfile.open(f.name, "w", dereference=True) as tar:
                tar.add(src_path, arcname=".")
            self.sandbox.files.upload(f.name, archive)
        self.execute(
            f"mkdir -p {shlex.quote(dest_path)} && tar -xf {archive} --no-same-owner -C {shlex.quote(dest_path)} && rm {archive}"
        )

    def get_template_vars(self) -> dict[str, Any]:
        return dict(self.config)


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


class ProbeModel(OpenAIChatModel):
    """Keeps every call's prompt and completion size. A turn re-sends the whole conversation, so the sum over calls
    is what sequence packing forwards for the rollout and the last call is about what tree packing forwards."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[tuple[int, int]] = []

    def _record_call(self, token_count) -> None:
        super()._record_call(token_count)
        self.calls.append((token_count.input_tokens, token_count.output_tokens))


class ProbeAgent(CCAgent):
    """The upstream agent plus the wall-clock limit verl's loop imposes on it from outside."""

    def __init__(self, *args, deadline: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline = deadline

    def query(self) -> dict:
        if time.monotonic() > self.deadline:
            raise LimitsExceeded("Trajectory timed out")
        return super().query()


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
        base_url: str,
        api_key: str,
        model: str,
        chat_template_kwargs: dict[str, Any],
        temperature: float,
        top_p: float,
        max_turn_tokens: int,
        max_observation_length: int,
        step_limit: int,
        agent_timeout: int,
        verify_timeout: int,
        judge_env: dict[str, str],
        flavor: str,
        transcripts_dir: Path,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.chat_template_kwargs = chat_template_kwargs
        self.temperature = temperature
        self.top_p = top_p
        self.max_turn_tokens = max_turn_tokens
        self.max_observation_length = max_observation_length
        self.step_limit = step_limit
        self.agent_timeout = agent_timeout
        self.verify_timeout = verify_timeout
        self.judge_env = judge_env
        self.flavor = flavor
        self.transcripts_dir = transcripts_dir

    def create(self, task: dict, sample: int) -> GeneralTaskSession:
        manifest = task["manifest"]
        # `general-agent-env-0:oss` in the dataset is `general-agent-env-0` on Docker Hub (see image-mapping.jsonl).
        # A dedicated sandbox, not a pooled one: a pooled sandbox is an unprivileged home on a shared host, so it
        # cannot write the `/work`, `/installed-agent` and `/logs` paths a task installs itself into.
        sandbox = Sandbox.create(
            image=f"{DOCKER_REPOSITORY}:{task['docker_image'].split(':')[0]}",
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
        return self.make_session(task=task, sandbox=sandbox, env=env, agent=agent)

    def make_model(self) -> ProbeModel:
        model_kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_turn_tokens,
        }
        if self.chat_template_kwargs:
            model_kwargs["extra_body"] = {"chat_template_kwargs": self.chat_template_kwargs}
        return ProbeModel(model_name=self.model, model_kwargs=model_kwargs)

    def make_session(self, **kwargs) -> GeneralTaskSession:
        return GeneralTaskSession(judge_env=self.judge_env, verify_timeout=self.verify_timeout, **kwargs)

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


# ============================================================================================================
# Probe
# ============================================================================================================


def run_one(factory: GeneralTaskSessionFactory, task: dict, sample: int) -> dict:
    t0 = time.perf_counter()
    result = {
        "instance_id": task["instance_id"],
        "sample": sample,
        "category": task["category"],
        "language": task["language"],
        "tier": task["tier"],
        "exit_status": None,
        "steps": 0,
        "tool_calls": {},
        "format_errors": 0,
        "blocked": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "calls": [],
        "final_answer": "",
        "reward": None,
        "reward_error": None,
        "judge_error": None,
        "items": {},
    }
    session = None
    try:
        session = factory.create(task, sample)
        exit_status, exit_message = session.run()
        agent = session.agent
        calls = Counter(
            call["function"]["name"]
            for message in agent.messages
            if message["role"] == "assistant"
            for call in message.get("tool_calls") or []
        )
        result.update(
            exit_status=exit_status,
            steps=agent._steps_taken,
            tool_calls=dict(calls),
            format_errors=sum(agent.tool_call_errors),
            blocked=len(agent.antihack.blocks),
            input_tokens=agent.model.token_stats.input_tokens,
            output_tokens=agent.model.token_stats.output_tokens,
            calls=agent.model.calls,
            final_answer=exit_message,
        )
        result.update(session.grade(exit_message))
    except (
        Exception
    ) as e:  # a sandbox that failed to start or died mid-rollout is a failed rollout, not a failed probe
        result.update(exit_status=type(e).__name__, error=str(e)[:500])
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
    result["wall_s"] = time.perf_counter() - t0
    print(
        f"[rollout] {result['instance_id']}#{sample} {result['exit_status']} steps={result['steps']} "
        f"reward={result['reward']} {result['reward_error'] or ''} {result['wall_s']:.0f}s",
        flush=True,
    )
    return result


def summarize(results: list[dict], tasks: list[dict]) -> None:
    scored = [r for r in results if r["reward"] is not None]
    masked = Counter(r["reward_error"] or r["exit_status"] for r in results if r["reward"] is None)
    print(f"\nrollouts: {len(results)}, scored: {len(scored)}, unscored: {dict(masked)}")
    print("exit statuses:", dict(Counter(r["exit_status"] for r in results)))
    if results:
        print(
            f"mean steps: {statistics.mean(r['steps'] for r in results):.1f}, "
            f"mean output tokens: {statistics.mean(r['output_tokens'] for r in results):.0f}, "
            f"mean wall: {statistics.mean(r['wall_s'] for r in results):.0f}s"
        )
        print("tool calls:", dict(sum((Counter(r["tool_calls"]) for r in results), Counter()).most_common(12)))
    summarize_packing(results)
    if not scored:
        return
    # Upstream trains on the binarized reward (`REWARD_BINARIZE_THRESHOLD=1.0`): a rollout passes only when every
    # rubric item does, so the pass rate is the number to read, the weighted score is the shape below it.
    rewards = [r["reward"] for r in scored]
    passed = [x >= 1 - 1e-6 for x in rewards]
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    bins = Counter(sum(x > edge for edge in edges) for x in rewards)
    labels = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1)", "1"]
    hist = ", ".join(f"{labels[k]}: {bins[k]}" for k in range(len(labels)) if bins[k])
    print(
        f"pass rate: {sum(passed)}/{len(scored)} = {statistics.mean(passed):.3f}; weighted score mean {statistics.mean(rewards):.3f}, histogram {hist}"
    )

    by_method: dict[str, list[float]] = defaultdict(list)
    methods = {}
    for task in tasks:
        for item in json.loads((task["task_dir"] / "verifier_meta.json").read_text())["items"]:
            methods[(task["instance_id"], item["id"])] = item.get("method", "?")
    for r in scored:
        for item_id, score in r["items"].items():
            by_method[methods.get((r["instance_id"], item_id), "gate")].append(score)
    print("rubric items by method:", {m: f"{statistics.mean(v):.2f} ({len(v)})" for m, v in sorted(by_method.items())})

    by_category: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_category[r["category"]].append(r["reward"])
    print("by category:", {c: f"{statistics.mean(v):.2f} ({len(v)})" for c, v in sorted(by_category.items())})

    by_task: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_task[r["instance_id"]].append(r["reward"])
    for instance_id, values in sorted(by_task.items()):
        n_pass = sum(x >= 1 - 1e-6 for x in values)
        print(
            f"  pass {n_pass}/{len(values)}  score {statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}  {instance_id}"
        )
    mixed = sum(0 < sum(x >= 1 - 1e-6 for x in v) < len(v) for v in by_task.values())
    spread = sum(max(v) - min(v) > 1e-6 for v in by_task.values())
    print(
        f"tasks with both a pass and a fail (GRPO signal on the binarized reward): {mixed}/{len(by_task)}; with any score spread: {spread}/{len(by_task)}"
    )


def summarize_packing(results: list[dict]) -> None:
    """What tree packing would save on these rollouts. Every turn re-sends the conversation, so sequence packing
    forwards the sum of all calls while a rollout folded into a prefix forest forwards about its final context; the
    samples of a task then share the opening prompt (system, tools, task) on top of that."""
    rollouts = [r for r in results if r["calls"]]
    if not rollouts:
        return
    forwarded = sum(sum(i + o for i, o in r["calls"]) for r in rollouts)
    context = sum(r["calls"][-1][0] + r["calls"][-1][1] for r in rollouts)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        by_task[r["instance_id"]].append(r)
    group_unique = sum(
        sum(r["calls"][-1][0] + r["calls"][-1][1] for r in group) - (len(group) - 1) * group[0]["calls"][0][0]
        for group in by_task.values()
    )
    print(
        f"packing: mean turns {statistics.mean(len(r['calls']) for r in rollouts):.1f}, "
        f"mean forwarded tokens {forwarded / len(rollouts):.0f} vs mean final context {context / len(rollouts):.0f}; "
        f"ratio within a rollout {forwarded / context:.2f}x, with the task's samples in one row {forwarded / group_unique:.2f}x"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="trl")
    p.add_argument("--output", default="probe_general.jsonl")
    p.add_argument("--n-tasks", type=int, default=32)
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--categories", default=None)  # comma-separated, e.g. accounting_audit_tax,legal_compliance
    p.add_argument("--language", default="en", choices=["en", "zh", "all"])
    p.add_argument("--max-inflight", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)  # upstream general.sh: 1.0, top_p 0.95
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-turn-tokens", type=int, default=4096)
    # Characters kept per tool result. mimoagent's 40000 is sized for MiMo's 262k context; at Qwen3's 40960 tokens
    # two parallel reads of a CSV would fill the whole window in one turn.
    p.add_argument("--max-observation-length", type=int, default=8000)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--step-limit", type=int, default=500)  # upstream s3k.yaml
    p.add_argument("--agent-timeout", type=int, default=1200)  # upstream TRAJECTORY_TIMEOUT
    p.add_argument("--verify-timeout", type=int, default=900)
    p.add_argument("--sandbox-flavor", default="cpu-basic")
    p.add_argument("--judge-url", default="https://router.huggingface.co/v1")
    p.add_argument("--judge-model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    p.add_argument("--judge-key", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    logging.getLogger("mimoagent").setLevel(logging.WARNING)
    transcripts_dir = Path(args.output).with_suffix(".transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(
        args.n_tasks,
        args.seed,
        set(args.categories.split(",")) if args.categories else None,
        None if args.language == "all" else args.language,
    )
    print(f"{len(tasks)} tasks: {dict(Counter(t['category'] for t in tasks))}", flush=True)

    factory = GeneralTaskSessionFactory(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        chat_template_kwargs={} if args.enable_thinking else {"enable_thinking": False},
        temperature=args.temperature,
        top_p=args.top_p,
        max_turn_tokens=args.max_turn_tokens,
        max_observation_length=args.max_observation_length,
        step_limit=args.step_limit,
        agent_timeout=args.agent_timeout,
        verify_timeout=args.verify_timeout,
        judge_env=judge_env(args.judge_url, args.judge_key, args.judge_model),
        flavor=args.sandbox_flavor,
        transcripts_dir=transcripts_dir,
    )

    jobs = [(task, sample) for task in tasks for sample in range(args.samples_per_task)]
    results = []
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool, open(args.output, "w") as out:
        for future in as_completed([pool.submit(run_one, factory, *job) for job in jobs]):
            result = future.result()
            results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
    summarize(results, tasks)


if __name__ == "__main__":
    main()
