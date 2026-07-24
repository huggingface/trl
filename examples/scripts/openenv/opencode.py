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
#     "trl",
#     "trackio",
#     "datasets",
#     "openenv-opencode @ git+https://github.com/meta-pytorch/OpenEnv.git#subdirectory=envs/opencode_env",
# ]
# ///

"""AsyncGRPO training of the real `opencode` coding agent (loop-owning) with a local subprocess sandbox.

`opencode` is a genuine external coding agent that owns its own tool loop. Each rollout runs it as a local process
(no container) via the `LocalSubprocessSandboxBackend` defined below, in `transparent_proxy` mode: an in-sandbox
proxy forwards the agent's calls to your vLLM server and captures per-turn `(token_ids, logprobs)`. TRL reads that
proxy trace, rebuilds training rows, scores the workspace with a held-out verifier, and trains with GRPO.

Task: competitive-coding problems from `agentica-org/DeepCoder-Preview-Dataset`. The agent writes `solution.py`
(reads stdin, prints stdout); the verifier runs it against the problem's HELD-OUT tests (never shown to the agent)
and returns a DENSE reward = fraction passed. `opencode_reward` then binarizes it and adds small degeneracy
penalties. This whole file is self-contained and every training-facing object is module-level (picklable), so the
rollout worker can pickle the factory + verifier into its spawned child process.

Requirements:
  - An OpenAI-compatible vLLM server (see below) reachable at `--vllm-url`.
  - Internet on this node the first time: `warmup()` installs the `opencode` CLI into a template dir once.
  - `pip install git+https://github.com/meta-pytorch/OpenEnv.git#subdirectory=envs/opencode_env`

Run (2 GPUs: vLLM on one, trainer on the other):

```sh
# Terminal 1 - serve the policy. Tool-calling + token-ids + NCCL weight-sync are all required.
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2 - train.
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/opencode.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --vllm-url http://localhost:8000
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from opencode_env import harness as oc_harness
from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.sandbox.base import ExecResult, SandboxHandle
from opencode_env.task import OpenCodeTask
from openenv.core.harness import ResourceSession, ResourceSessionFactory, VerifyResult
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import (
    HarnessRolloutOutcome,
    HarnessRolloutWorker,
    TraceEntry,
    has_tool_call,
)


# ============================================================================================================
# Local subprocess sandbox backend
# ------------------------------------------------------------------------------------------------------------
# OpenEnv's opencode harness only ships an E2B (cloud) backend, and a cloud sandbox can't reach a local vLLM.
# The `SandboxBackend` protocol is small, so we run opencode + its proxy as local processes on this node. The
# harness bakes the prefix `/home/user` into several paths, so each sandbox REMAPS that prefix to its own dir and
# callers pass `OpenCodeConfig(sandbox_home="/home/user")` so config-driven paths funnel through the same remap.
# ============================================================================================================

_OPENCODE_INSTALL = "curl -fsSL https://opencode.ai/install | bash -s -- --no-modify-path"


class LocalBgJob:
    """A background process (the opencode agent or its proxy) running directly on the node."""

    def __init__(self, popen: subprocess.Popen):
        self._p = popen

    @property
    def pid(self) -> int:
        return self._p.pid

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self._p.wait(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(str(e)) from e

    def kill(self) -> None:
        # Kill the whole process GROUP: opencode spawns a tree (node -> bash -> python); SIGTERM to only the parent
        # orphans the children, which pile up across rollouts. `start_bg` launches each job in its own session.
        if self._p.poll() is not None:
            return
        try:
            pgid = os.getpgid(self._p.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            self._p.wait(timeout=5)
        except (subprocess.TimeoutExpired, Exception):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class LocalSandboxHandle:
    """One local 'sandbox' = a real directory on the node. The harness's hardcoded `/home/user` prefix is remapped
    to this directory in every command and path, and `$HOME` points at it. `kill()` removes the directory."""

    def __init__(
        self,
        root: str,
        *,
        home_alias: str = "/home/user",
        base_env: dict[str, str] | None = None,
        cleanup: bool = False,
    ):
        self._root = root
        self._alias = home_alias
        self._cleanup = cleanup
        self._env = {**os.environ, "HOME": root, **(base_env or {})}
        self._bg: list[LocalBgJob] = []

    @property
    def sandbox_id(self) -> str:
        return self._root

    def _remap(self, s: str | None) -> str | None:
        return s if s is None else s.replace(self._alias, self._root)

    def _run_env(self, envs: dict[str, str] | None) -> dict[str, str]:
        return {**self._env, **(envs or {})}

    def exec(self, cmd: str, *, envs=None, cwd=None, timeout: float | None = 60) -> ExecResult:
        try:
            p = subprocess.run(
                ["bash", "-lc", self._remap(cmd)],
                cwd=self._remap(cwd) or self._root,
                env=self._run_env(envs),
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            return ExecResult(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired as e:
            return ExecResult(exit_code=124, stdout=e.stdout or "", stderr=f"timeout after {timeout}s")

    def start_bg(self, cmd: str, *, envs=None, cwd=None) -> LocalBgJob:
        # stdin=/dev/null so the agent (and any `python solution.py` it runs) reads EOF instead of blocking forever.
        p = subprocess.Popen(
            ["bash", "-lc", self._remap(cmd)],
            cwd=self._remap(cwd) or self._root,
            env=self._run_env(envs),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group so kill() reaps the whole opencode tree
        )
        job = LocalBgJob(p)
        self._bg.append(job)
        return job

    def write_text(self, path: str, content: str) -> None:
        path = self._remap(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    def read_text(self, path: str) -> str:
        return Path(self._remap(path)).read_text()

    def exists(self, path: str) -> bool:
        return Path(self._remap(path)).exists()

    def kill(self) -> None:
        for job in self._bg:
            try:
                job.kill()
            except Exception:
                pass
        self._bg.clear()
        if self._cleanup:
            shutil.rmtree(self._root, ignore_errors=True)


class LocalSubprocessSandboxBackend:
    """Produces per-rollout `LocalSandboxHandle`s, each in its own `uuid` dir hardlink-cloned from a template that
    has opencode pre-installed (`warmup()`), so concurrent sandboxes never share state and never re-install."""

    def __init__(self, root: str, *, home_alias: str = "/home/user"):
        self._root = root
        self._alias = home_alias
        self._template = os.path.join(root, "_template")

    def warmup(self) -> None:
        """Install opencode ONCE into the template dir (run in the parent, before rollouts spawn)."""
        marker = os.path.join(self._template, ".opencode", "bin", "opencode")
        if os.path.exists(marker):
            return
        os.makedirs(self._template, exist_ok=True)
        subprocess.run(
            ["bash", "-lc", _OPENCODE_INSTALL],
            env={**os.environ, "HOME": self._template},
            check=True,
            timeout=400,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def create(self, *, timeout_s: int = 900, envs=None, metadata=None) -> LocalSandboxHandle:
        name = (metadata or {}).get("episode_id") or uuid.uuid4().hex
        sdir = os.path.join(self._root, name)
        shutil.rmtree(sdir, ignore_errors=True)
        os.makedirs(sdir, exist_ok=True)
        if os.path.isdir(self._template):
            subprocess.run(["cp", "-al", f"{self._template}/.", f"{sdir}/"], check=True)  # hardlink-clone
        for sub in ("workdir", "task", "logs/agent", "logs/verifier", ".config/opencode"):
            d = os.path.join(sdir, sub)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        return LocalSandboxHandle(sdir, home_alias=self._alias, base_env=envs, cleanup=True)


# ============================================================================================================
# Dataset + held-out verifier
# ============================================================================================================

DATASET = "agentica-org/DeepCoder-Preview-Dataset"
DATASET_CONFIG = "primeintellect"  # carries 11-93 stdin/stdout tests per problem
N_TESTS_EVAL = 12  # cap held-out tests scored per rollout (bounds verify latency); dense reward stays fine-grained
PER_TEST_TIMEOUT = 6


def _instruction_id(instruction: str) -> str:
    return hashlib.sha1(instruction.encode()).hexdigest()


def _clean_problem(problem: str) -> str:
    # DeepCoder wraps problems in boilerplate that fights our tool-writing instruction ("...return the code."), which
    # nudges a weak model to dump code in its reply instead of writing solution.py. Strip it so our instruction leads.
    text = problem.strip()
    prefix = "Solve the following coding problem using the programming language python:"
    if text.startswith(prefix):
        text = text[len(prefix) :].strip()
    cut = len(text)
    for marker in ("The input will be", "Now solve the problem", "Now solve this problem"):
        idx = text.rfind(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def _coding_instruction(problem: str) -> str:
    # Multi-turn loop: write -> run on the example cases -> read feedback -> fix -> repeat. The examples in the problem
    # are the feedback signal; the held-out tests (the reward) stay hidden.
    problem = _clean_problem(problem)
    return (
        "Solve the problem below by writing a Python program to `solution.py` in the current working directory "
        "(it does not exist yet, so create it with the `write` tool). The program must read its input from standard "
        "input (stdin) and write ONLY the required answer to standard output (stdout).\n"
        "Then TEST it against the example cases shown in the problem: run your program on each example input with "
        "bash, e.g. `printf '<example input>' | python3 solution.py`, and compare its output to the expected output. "
        "If an example fails or the program errors, use the `edit` tool to fix `solution.py`, then run it again. "
        "Keep going until your program passes all the provided examples, then stop.\n\n"
        f"PROBLEM:\n{problem}"
    )


# Runner executed INSIDE the sandbox: run solution.py against each held-out test with a hard time/mem/output cap,
# compare normalized stdout, print `SCORE: <fraction>`. Whitespace-insensitive match (competitive judges usually are).
_RUNNER_SRC = r"""
import json, subprocess, sys

PER_TEST_TIMEOUT = {per_test_timeout}

def norm(s):
    lines = [ln.rstrip() for ln in (s or "").replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)

def ok(got, exp):
    if norm(got) == norm(exp):
        return True
    return (got or "").split() == (exp or "").split()

tests = json.load(open("_tests.json"))
passed = 0
for t in tests:
    cmd = "ulimit -v 2000000 2>/dev/null; timeout %d python3 solution.py 2>/dev/null | head -c 2000000" % PER_TEST_TIMEOUT
    try:
        p = subprocess.run(["bash", "-c", cmd], input=t.get("input", ""), capture_output=True, text=True,
                           timeout=PER_TEST_TIMEOUT + 5)
        if ok(p.stdout, t.get("output", "")):
            passed += 1
    except Exception:
        pass
print("SCORE: %.6f" % (passed / len(tests) if tests else 0.0))
"""


def _run_dense_tests(sandbox: SandboxHandle, tests: list[dict[str, str]]) -> float:
    """Run the sandbox's `solution.py` against `tests`; return the fraction passed."""
    if not tests or not sandbox.exists("/home/user/workdir/solution.py"):
        return 0.0
    tests = tests[:N_TESTS_EVAL]
    sandbox.write_text("/home/user/workdir/_tests.json", json.dumps(tests))
    sandbox.write_text("/home/user/workdir/_run_tests.py", _RUNNER_SRC.format(per_test_timeout=PER_TEST_TIMEOUT))
    r = sandbox.exec("cd /home/user/workdir && python3 _run_tests.py", timeout=PER_TEST_TIMEOUT * len(tests) + 30)
    for line in (r.stdout or "").splitlines():
        if line.startswith("SCORE:"):
            return float(line.split(":", 1)[1].strip())
    return 0.0


class DeepCoderStdinVerifier:
    """Dense stdin/stdout verifier. Holds the held-out test map (keyed by `sha1(instruction)`) so it survives the
    pickle into the rollout child. `session.verify(...)` calls it as `verifier(sandbox, task)`."""

    def __init__(self, tests_by_id: dict[str, list[dict[str, str]]]):
        self._tests_by_id = tests_by_id

    def __call__(self, sandbox: SandboxHandle, task: OpenCodeTask) -> VerifyResult:
        tests = self._tests_by_id.get(_instruction_id(task.instruction))
        return VerifyResult(env_reward=_run_dense_tests(sandbox, tests or []), done=True)


def build_dataset(n_prompts: int, seed: int) -> tuple[list[dict], dict[str, list[dict[str, str]]]]:
    """Return `(rows, tests_by_id)`: the prompt rows (problem statement only) and the held-out test map for the
    verifier. Tests are NOT put in the prompt - the agent only sees the statement (which includes sample cases)."""
    rows = list(load_dataset(DATASET, DATASET_CONFIG, split="train"))
    random.Random(seed).shuffle(rows)
    out: list[dict] = []
    tests_by_id: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        if len(out) >= n_prompts:
            break
        raw = r["tests"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        tests = [
            {"input": t.get("input", ""), "output": t.get("output", "")}
            for t in raw
            if t.get("type", "stdin_stdout") == "stdin_stdout" and t.get("output") is not None
        ]
        if len(tests) < 3:  # need enough tests for a meaningful dense fraction
            continue
        instruction = _coding_instruction(r["problem"])
        tests_by_id[_instruction_id(instruction)] = tests[:N_TESTS_EVAL]
        out.append({"prompt": [{"role": "user", "content": instruction}]})
    return out, tests_by_id


# ============================================================================================================
# opencode session factory (local sandbox + per-session proxy port)
# ============================================================================================================


class OpencodeTaskFactory(ResourceSessionFactory):
    """Adapts the worker's `create(prompt=messages, seed, episode_id)` onto `OpenCodeSessionFactory`, which wants an
    `OpenCodeTask`: pull the instruction out of the last user message."""

    def __init__(self, inner: OpenCodeSessionFactory):
        self._inner = inner

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> ResourceSession:
        instruction = task[-1]["content"] if isinstance(task, list) and task else str(task)
        return self._inner.create(instruction, seed=seed, episode_id=episode_id)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FreePortOpenCodeSessionFactory(OpenCodeSessionFactory):
    """Same as `OpenCodeSessionFactory` but binds the in-sandbox proxy to a free port per session instead of the
    hardcoded `_PROXY_PORT = 7000`, so several opencode sandboxes can run at once on one node. Mirrors OpenEnv's
    `_start_proxy` exactly except for the port."""

    def _start_proxy(self, sandbox):
        port = _free_port()
        trace_path = oc_harness._PROXY_TRACE_PATH
        log_path = oc_harness._PROXY_LOG_PATH
        if not sandbox.exists("/home/user/proxy/interception.py"):
            self._exec_with_retry(
                sandbox,
                "pip install --quiet 'fastapi>=0.104' 'uvicorn[standard]>=0.24' 'httpx>=0.27' 2>&1 | tail -20",
                timeout=180,
                attempts=3,
                backoff_s=2.0,
                label="proxy deps install",
            )
            sandbox.write_text("/home/user/proxy/interception.py", oc_harness._PROXY_SOURCE_PATH.read_text())
            sandbox.write_text("/home/user/proxy/__init__.py", "")

        proxy_args = [
            "python", "interception.py", "--upstream-url", self._config.base_url,
            "--trace", trace_path, "--port", str(port),
            "--top-logprobs", str(self._config.proxy_top_logprobs),
        ]  # fmt: skip
        if self._config.proxy_max_tokens_cap is not None:
            proxy_args += ["--max-tokens-cap", str(self._config.proxy_max_tokens_cap)]
        if self._config.proxy_disable_thinking:
            proxy_args.append("--disable-thinking")
        if self._config.model:
            proxy_args += ["--model-override", self._config.model]

        quoted = " ".join(shlex.quote(a) for a in proxy_args)
        proxy_cmd = f"cd /home/user/proxy && {quoted} > {shlex.quote(log_path)} 2>&1"
        proxy_job = sandbox.start_bg(proxy_cmd, envs={"OPENCODE_UPSTREAM_API_KEY": self._config.api_key})

        for _ in range(120):
            if sandbox.exec(f"curl -sf http://127.0.0.1:{port}/healthz", timeout=5).exit_code == 0:
                break
            time.sleep(0.5)
        else:
            log = ""
            try:
                log = sandbox.read_text(log_path)
            except Exception:
                pass
            proxy_job.kill()
            raise RuntimeError(f"proxy did not start on :{port}\n{log[-2000:]}")

        return proxy_job, f"http://127.0.0.1:{port}/v1", trace_path


def build_factory(sandbox_root: str, vllm_url: str, model: str, tests_by_id: dict) -> OpencodeTaskFactory:
    config = OpenCodeConfig(
        provider="openai_compatible",
        base_url=f"{vllm_url}/v1",
        model=model,  # proxy --model-override forces this exact id on upstream requests
        sandbox_home="/home/user",  # remapped to each sandbox's real dir by LocalSandboxHandle
        agent_timeout_s=180.0,  # bounds edit/bash-loop blowups; legit solves finish in <90s
        disabled_tools=["webfetch", "question", "task"],  # no web, no user, no sub-agents
        run_format="json",
    )
    backend = LocalSubprocessSandboxBackend(sandbox_root)
    backend.warmup()  # install opencode ONCE (parent, before rollouts)
    inner = FreePortOpenCodeSessionFactory(
        config=config,
        sandbox_backend=backend,
        mode="transparent_proxy",  # in-sandbox proxy captures completion_token_ids + per_token_logps
        verifier=DeepCoderStdinVerifier(tests_by_id),
    )
    return OpencodeTaskFactory(inner)


# ============================================================================================================
# Reward + turn-selection policy (application-owned; passed to the worker as hooks)
# ============================================================================================================


def opencode_reward(outcome: HarnessRolloutOutcome) -> float | None:
    """Binary terminal verifier + degeneracy penalties. Long-horizon credit is carried by the terminal reward,
    propagated to every trained token through the group-relative advantage.

      - unscorable rollout -> None (dropped from the group baseline)
      - never ran its code (no `bash`) -> -0.1 (kills blind-write / prose-dump / give-up)
      - else BINARY base: all held-out tests pass -> 1.0; timed out or failed -> 0.0
      - minus a step penalty for tool calls beyond a budget (bounds runaway edit/bash loops), capped at 0.5
    """
    step_budget, step_penalty, step_penalty_cap = 20, 0.03, 0.5
    frac = outcome.env_reward
    if frac is None:
        return None
    if outcome.tool_calls_by_name.get("bash", 0) == 0:
        return -0.1
    base = 0.0 if outcome.timed_out else (1.0 if frac >= 1.0 - 1e-9 else 0.0)
    over = max(0, outcome.tool_call_count - step_budget)
    return base - min(step_penalty_cap, step_penalty * over)


def opencode_agent_turns(trace: list[TraceEntry]) -> list[TraceEntry]:
    """`agent_turn_fn`: keep only the REAL agent turns. opencode fires extra LLM calls for its own bookkeeping (a
    title generator, a context summarizer) either without tools or with a different system prompt; those are a
    different task and must not be trained/scored. The agent loop reuses ONE tool-enabled system prompt, so anchor
    on the first tool-enabled turn's system prompt and keep only matching entries."""

    def system_of(messages):
        return next((m.get("content") for m in messages if m.get("role") == "system"), None)

    primary = None
    for entry in trace:
        request = entry.get("request") or {}
        if request.get("messages") and request.get("tools"):
            primary = system_of(request["messages"])
            break
    return [
        entry
        for entry in trace
        if (request := entry.get("request") or {}).get("messages")
        and request.get("tools")
        and system_of(request["messages"]) == primary
    ]


# ============================================================================================================
# Training
# ============================================================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument(
        "--num-generations", type=int, default=8
    )  # >1 gives within-group pass/fail split -> nonzero advantage
    p.add_argument("--max-inflight", type=int, default=8)  # concurrent rollouts (each its own sandbox + proxy port)
    p.add_argument("--max-completion-length", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)  # lower -> fresher rollouts -> ratios near 1 (more stable)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="async_grpo_opencode")
    p.add_argument("--project", default="opencode")
    p.add_argument("--trackio-space-id", default=None)  # optional: host the trackio dashboard on a HF Space
    p.add_argument("--sandbox-root", default=None)  # where per-rollout sandbox dirs live (default: a fresh tempdir)
    args = p.parse_args()

    sandbox_root = args.sandbox_root or tempfile.mkdtemp(prefix="trl_opencode_")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows, tests_by_id = build_dataset(n_prompts=args.n_prompts, seed=args.seed)
    dataset = Dataset.from_list(rows)

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        save_strategy="no",
        per_device_train_batch_size=4,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        max_staleness=args.max_staleness,
        vllm_server_base_url=args.vllm_url,
        report_to="trackio",
        project=args.project,
        trackio_space_id=args.trackio_space_id,
        log_completions=True,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=build_factory(sandbox_root, args.vllm_url, args.model, tests_by_id),
        harness_adapter=None,  # loop-owning: opencode runs its own loop; TRL reads the proxy trace
        rollout_reward_fn=opencode_reward,  # reward policy (binary verifier + degeneracy penalties)
        train_turn_fn=has_tool_call,  # coding agent: reinforce only action turns, not prose
        agent_turn_fn=opencode_agent_turns,  # drop opencode's title/summarizer aux calls from the trace
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # reward comes from the harness verifier via rollout_reward_fn, not reward_funcs
        processing_class=tokenizer,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        fork_threshold_tokens=1024,
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
    )
    trainer.train()


if __name__ == "__main__":
    main()
