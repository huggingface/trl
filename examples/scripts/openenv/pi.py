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
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "transformers>=5.2",
#     "kernels",
#     "bitsandbytes",
#     "trackio",
#     "datasets",
#     "huggingface_hub>=1.22",
#     "openenv @ git+https://github.com/sergiopaniego/OpenEnv.git@pi-hf-sandbox-backend",
#     "openenv-opencode-env @ git+https://github.com/sergiopaniego/OpenEnv.git@pi-hf-sandbox-backend#subdirectory=envs/opencode_env",
#     "openenv-pi-env @ git+https://github.com/sergiopaniego/OpenEnv.git@pi-hf-sandbox-backend#subdirectory=envs/pi_env",
# ]
# ///

"""AsyncGRPO training of the real Pi coding agent (loop-owning) with a LOCAL subprocess sandbox.

Local analog of pi_hf_sandbox.py: everything runs on one node, no remote sandbox and no tunnel. Pi speaks the
OpenAI API natively (no translation shim), so each rollout runs Pi as a local process behind one in-process
helper: Pi -> interception proxy (captures token_ids + logprobs) -> your localhost vLLM. Pi owns its own tool
loop. TRL reads the proxy trace, rebuilds training rows, scores the workspace with a held-out verifier, and
trains with GRPO.

Task: competitive-coding problems from `agentica-org/DeepCoder-Preview-Dataset`. The agent writes `solution.py`
(reads stdin, prints stdout); the verifier runs it against the problem's HELD-OUT tests (never shown to the
agent) and returns a DENSE reward = fraction passed. `pi_reward` then binarizes it and adds small degeneracy
penalties. This whole file is self-contained and every training-facing object is module-level (picklable), so
the rollout worker can pickle the factory + verifier into its spawned child process.

Pi's runtime derives every path from an ABSOLUTE `sandbox_home` (`.pi-npm/bin/pi`, `.node`, `workdir`, `.pi`,
`proxy`, ...). Rather than write into a real host `/home/user`, we set `sandbox_home` to the alias `/home/user`
and REMAP that prefix (in commands, paths, cwd, and harness-supplied env values) to a per-rollout tempdir.
`warmup()` installs Node 22 + the Pi CLI ONCE into a template dir (installed at the template's real paths), and
`create()` hardlink-clones the template per rollout.

Requirements:
  - An OpenAI-compatible vLLM server (see below) reachable at `--vllm-url` on localhost.
  - Internet on this node the first time: `warmup()` installs Pi into a template dir once.

Run (2 GPUs: vLLM on one, trainer on the other):

```sh
# Terminal 1 - serve the policy. Tool-calling + token-ids + NCCL weight-sync are all required.
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2 - train. The proxy runs as a local subprocess on this node.
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/pi.py \
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

import pi_env.harness as pi_harness
from datasets import Dataset, load_dataset
from opencode_env.sandbox.base import ExecResult, SandboxHandle
from openenv.core.harness import ResourceSession, ResourceSessionFactory, VerifyResult
from pi_env.config import PiConfig
from pi_env.harness import PiSessionFactory
from pi_env.pi_runtime import (
    build_install_cmd,
    proxy_dir,
    proxy_log_path,
    proxy_source_path,
    proxy_trace_path,
)
from pi_env.task import PiTask
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import (
    HarnessRolloutOutcome,
    HarnessRolloutWorker,
    has_tool_call,
)


# Alias baked into `sandbox_home`. Every Pi path hangs off this prefix and is remapped, per rollout, to the
# sandbox's real tempdir by LocalSandboxHandle. `/home/user` (not `/root`) so the remap can't collide with real
# host paths on a node whose user home differs.
SANDBOX_HOME = "/home/user"
WORKDIR = f"{SANDBOX_HOME}/workdir"
# Pi's built-in tool names; restrict to the coding-relevant ones (no web / sub-agents).
ALLOWED_TOOLS = ["bash", "read", "edit", "write", "grep", "ls"]


# ============================================================================================================
# Local subprocess sandbox backend
# ------------------------------------------------------------------------------------------------------------
# OpenEnv's Pi harness only ships remote backends, and a remote sandbox can't reach a local vLLM. The
# `SandboxHandle` protocol is small, so we run Pi + its proxy as local processes on this node. The harness
# derives every path from `config.sandbox_home`, so each sandbox REMAPS that prefix (the alias `/home/user`) to
# its own dir, and callers pass `PiConfig(sandbox_home="/home/user")` so config-driven paths funnel through the
# same remap.
# ============================================================================================================


class LocalBgJob:
    """A background process (the Pi agent or its proxy) running directly on the node."""

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
        # Kill the whole process GROUP: Pi spawns a tree (node -> bash -> python); SIGTERM to only the parent
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
    """One local 'sandbox' = a real directory on the node. The harness's `sandbox_home` prefix (the alias) is
    remapped to this directory in every command, path, cwd, and harness-supplied env value, and `$HOME` points at
    it. `kill()` removes the directory."""

    def __init__(
        self,
        root: str,
        *,
        home_alias: str = SANDBOX_HOME,
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
        # Remap ONLY the harness-supplied env values (e.g. PI_CODING_AGENT_DIR=/home/user/.pi/agent). The inherited
        # environment is left untouched so a real `/home/user` on the host is never rewritten.
        extra = {k: self._remap(v) for k, v in (envs or {}).items()}
        return {**self._env, **extra}

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
            start_new_session=True,  # own process group so kill() reaps the whole Pi tree
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
    has Pi pre-installed (`warmup()`), so concurrent sandboxes never share state and never re-install.

    The install writes into the template at the template's REAL paths (a `PiConfig` copy whose `sandbox_home` is
    the template dir), so the on-disk layout matches what the runtime expects after the alias remap:
    `.pi-npm/bin/pi`, `.node`, `.node -> node-vX`, etc."""

    def __init__(self, root: str, config: PiConfig, *, home_alias: str = SANDBOX_HOME):
        self._root = root
        self._config = config
        self._alias = home_alias
        self._template = os.path.join(root, "_template")

    def warmup(self) -> None:
        """Install Node 22 + Pi ONCE into the template dir (run in the parent, before rollouts spawn)."""
        marker = os.path.join(self._template, ".pi-npm", "bin", "pi")
        if os.path.exists(marker):
            return
        os.makedirs(self._template, exist_ok=True)
        # Point sandbox_home at the template's real path so build_install_cmd writes the whole layout INTO the
        # template (Node tarball, the `.node -> node-vX` symlink, and the `.pi-npm` npm prefix). After the
        # per-rollout `cp -al` clone, the alias remap makes those same files resolve under each sandbox dir.
        install_config = self._config.model_copy(update={"sandbox_home": self._template})
        subprocess.run(
            ["bash", "-lc", build_install_cmd(install_config)],
            env={**os.environ, "HOME": self._template},
            check=True,
            timeout=600,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def create(self, *, timeout_s: int = 900, envs=None, metadata=None) -> LocalSandboxHandle:
        name = (metadata or {}).get("episode_id") or uuid.uuid4().hex
        sdir = os.path.join(self._root, name)
        shutil.rmtree(sdir, ignore_errors=True)
        os.makedirs(sdir, exist_ok=True)
        if os.path.isdir(self._template):
            # `cp -al` hardlink-clones files and preserves the `.node` symlink (its target is the template's
            # absolute node dir, which persists), so Node + Pi are shared read-only across sandboxes.
            subprocess.run(["cp", "-al", f"{self._template}/.", f"{sdir}/"], check=True)
        # Reset the Pi per-rollout dirs (leave the install: .node, node-vX, .pi-npm).
        for sub in ("workdir", "task", "logs/agent", "logs/verifier", ".pi/agent", "proxy"):
            d = os.path.join(sdir, sub)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        return LocalSandboxHandle(sdir, home_alias=self._alias, base_env=envs, cleanup=True)


# ============================================================================================================
# Dataset + held-out verifier (identical to pi_hf_sandbox.py)
# ============================================================================================================

DATASET = "agentica-org/DeepCoder-Preview-Dataset"
DATASET_CONFIG = "primeintellect"
N_TESTS_EVAL = 12
PER_TEST_TIMEOUT = 6


def _instruction_id(instruction: str) -> str:
    return hashlib.sha1(instruction.encode()).hexdigest()


def _clean_problem(problem: str) -> str:
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
    problem = _clean_problem(problem)
    return (
        "Solve the problem below by writing a Python program to `solution.py` in the current working directory "
        "(it does not exist yet, so create it). The program must read its input from standard input (stdin) and "
        "write ONLY the required answer to standard output (stdout).\n"
        "Then TEST it against the example cases shown in the problem: run your program on each example input with "
        "bash, e.g. `printf '<example input>' | python3 solution.py`, and compare its output to the expected output. "
        "If an example fails or the program errors, fix `solution.py` and run it again. "
        "Keep going until your program passes all the provided examples, then stop.\n\n"
        f"PROBLEM:\n{problem}"
    )


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
    if not tests or not sandbox.exists(f"{WORKDIR}/solution.py"):
        return 0.0
    tests = tests[:N_TESTS_EVAL]
    sandbox.write_text(f"{WORKDIR}/_tests.json", json.dumps(tests))
    sandbox.write_text(f"{WORKDIR}/_run_tests.py", _RUNNER_SRC.format(per_test_timeout=PER_TEST_TIMEOUT))
    r = sandbox.exec(f"cd {WORKDIR} && python3 _run_tests.py", timeout=PER_TEST_TIMEOUT * len(tests) + 30)
    for line in (r.stdout or "").splitlines():
        if line.startswith("SCORE:"):
            return float(line.split(":", 1)[1].strip())
    return 0.0


class DeepCoderStdinVerifier:
    def __init__(self, tests_by_id: dict[str, list[dict[str, str]]]):
        self._tests_by_id = tests_by_id

    def __call__(self, sandbox: SandboxHandle, task: PiTask) -> VerifyResult:
        tests = self._tests_by_id.get(_instruction_id(task.instruction))
        return VerifyResult(env_reward=_run_dense_tests(sandbox, tests or []), done=True)


def build_dataset(n_prompts: int, seed: int) -> tuple[list[dict], dict[str, list[dict[str, str]]]]:
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
        if len(tests) < 3:
            continue
        instruction = _coding_instruction(r["problem"])
        tests_by_id[_instruction_id(instruction)] = tests[:N_TESTS_EVAL]
        out.append({"prompt": [{"role": "user", "content": instruction}]})
    return out, tests_by_id


# ============================================================================================================
# Pi session factory (local sandbox + per-session proxy port)
# ============================================================================================================


class PiTaskFactory(ResourceSessionFactory):
    """Adapt the worker's `create(prompt=messages, seed, episode_id)` onto `PiSessionFactory`."""

    def __init__(self, inner: PiSessionFactory):
        self._inner = inner

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> ResourceSession:
        instruction = task[-1]["content"] if isinstance(task, list) and task else str(task)
        return self._inner.create(instruction, seed=seed, episode_id=episode_id)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FreePortPiSessionFactory(PiSessionFactory):
    """Same as `PiSessionFactory` but binds the proxy to a free port per session instead of the hardcoded
    `_PROXY_PORT = 7000`, so several Pi sandboxes can run at once on one node. Mirrors OpenEnv's `_start_proxy`
    exactly except for the port. The proxy runs as a local subprocess under the SYSTEM python (the trainer env
    already has fastapi/uvicorn/httpx via vLLM), so no in-template pip install is needed."""

    def _start_proxy(self, sandbox):
        port = _free_port()
        deps_present = sandbox.exec("python -c 'import fastapi, uvicorn, httpx'", timeout=15).exit_code == 0
        if not deps_present:
            self._exec_with_retry(
                sandbox,
                "set -o pipefail && pip install --quiet 'fastapi>=0.104' "
                "'uvicorn[standard]>=0.24' 'httpx>=0.27' 2>&1 | tail -20",
                timeout=180,
                attempts=3,
                backoff_s=2.0,
                label="proxy deps install",
            )

        if not sandbox.exists(proxy_source_path(self._config)):
            sandbox.write_text(proxy_source_path(self._config), pi_harness._PROXY_SOURCE_PATH.read_text())
            sandbox.write_text(f"{proxy_dir(self._config)}/__init__.py", "")

        proxy_args = [
            "python", "interception.py", "--upstream-url", self._config.base_url,
            "--trace", proxy_trace_path(self._config), "--port", str(port),
            "--top-logprobs", str(self._config.proxy_top_logprobs),
        ]  # fmt: skip
        if self._config.proxy_max_tokens_cap is not None:
            proxy_args += ["--max-tokens-cap", str(self._config.proxy_max_tokens_cap)]
        if self._config.proxy_disable_thinking:
            proxy_args.append("--disable-thinking")
        if self._config.model:
            proxy_args += ["--model-override", self._config.model]

        quoted = " ".join(shlex.quote(a) for a in proxy_args)
        proxy_cmd = (
            f"cd {shlex.quote(proxy_dir(self._config))} && {quoted} > {shlex.quote(proxy_log_path(self._config))} 2>&1"
        )
        proxy_job = sandbox.start_bg(proxy_cmd, envs={"UPSTREAM_API_KEY": self._config.api_key})

        for _ in range(120):
            if sandbox.exec(f"curl -sf http://127.0.0.1:{port}/healthz", timeout=5).exit_code == 0:
                break
            time.sleep(0.5)
        else:
            log = ""
            try:
                log = sandbox.read_text(proxy_log_path(self._config))
            except Exception:
                pass
            proxy_job.kill()
            raise RuntimeError(f"proxy did not start on :{port}\n{log[-2000:]}")

        return proxy_job, f"http://127.0.0.1:{port}/v1", proxy_trace_path(self._config)


def build_factory(sandbox_root: str, vllm_url: str, model: str, tests_by_id: dict) -> PiTaskFactory:
    config = PiConfig(
        base_url=f"{vllm_url}/v1",  # the in-sandbox proxy forwards here (localhost vLLM)
        model=model,  # proxy --model-override forces this exact id upstream
        sandbox_home=SANDBOX_HOME,  # remapped to each sandbox's real dir by LocalSandboxHandle
        agent_timeout_s=600.0,
        tools=ALLOWED_TOOLS,
        # Cap forwarded max_tokens well below max_model_len so Pi's large, turn-growing prompt + max_tokens never
        # exceeds the vLLM context window (which 400s the request).
        proxy_max_tokens_cap=8192,
    )
    backend = LocalSubprocessSandboxBackend(sandbox_root, config)
    backend.warmup()  # install Node 22 + Pi ONCE (parent, before rollouts)
    inner = FreePortPiSessionFactory(
        config=config,
        sandbox_backend=backend,
        mode="transparent_proxy",  # proxy captures completion_token_ids + per_token_logps; Pi speaks OpenAI directly
        verifier=DeepCoderStdinVerifier(tests_by_id),
    )
    return PiTaskFactory(inner)


# ============================================================================================================
# Reward (application-owned hook)
# ============================================================================================================


def pi_reward(outcome: HarnessRolloutOutcome) -> float | None:
    """Binary terminal verifier + degeneracy penalties (mirror of opencode_reward, Pi tool names).

    - unscorable rollout -> None
    - never ran its code (no `bash`) -> -0.1
    - else BINARY: all held-out tests pass -> 1.0; else 0.0
    - minus a step penalty for tool calls beyond a budget, capped at 0.5
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


# ============================================================================================================
# Training
# ============================================================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--num-generations", type=int, default=8)  # >1 gives within-group pass/fail split
    p.add_argument("--max-inflight", type=int, default=8)  # concurrent rollouts (each its own sandbox + port)
    p.add_argument("--max-completion-length", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)  # lower -> fresher rollouts -> ratios near 1
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="async_grpo_pi")
    p.add_argument("--project", default="pi")
    p.add_argument("--trackio-space-id", default=None)  # optional: host the trackio dashboard on a HF Space
    p.add_argument("--sandbox-root", default=None)  # where per-rollout sandbox dirs live (default: a fresh tempdir)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default=None)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()

    sandbox_root = args.sandbox_root or tempfile.mkdtemp(prefix="trl_pi_")
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
        # Off: the completions printer can't render Pi's multi-turn trace
        # (the "completion" is a list of messages, not a string) and crashes the step.
        log_completions=False,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=build_factory(sandbox_root, args.vllm_url, args.model, tests_by_id),
        harness_adapter=None,  # loop-owning: Pi runs its own loop; TRL reads the proxy trace
        rollout_reward_fn=pi_reward,
        train_turn_fn=has_tool_call,  # reinforce only action turns, not prose
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
        log_completions=False,
        num_completions_to_print=0,
    )

    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
    )
    trainer.train()
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
