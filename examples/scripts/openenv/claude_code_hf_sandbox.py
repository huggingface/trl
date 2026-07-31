# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# /// script
# dependencies = [
#     "trl",
#     "trackio",
#     "datasets",
#     "huggingface_hub>=1.22",
#     "openenv @ git+https://github.com/sergiopaniego/OpenEnv.git@add-claude-code-env",
#     "openenv-claude-code-env @ git+https://github.com/sergiopaniego/OpenEnv.git@add-claude-code-env#subdirectory=envs/claude_code_env",
# ]
# ///

"""AsyncGRPO training of the real Claude Code agent (loop-owning) in REMOTE Hugging Face sandboxes.

Same loop-owning path as opencode_hf_sandbox.py, for the Claude Code CLI. Claude Code speaks the Anthropic
Messages API only, so each rollout runs Claude Code in a remote HF sandbox behind an in-sandbox translation
shim: Claude Code -> shim (Anthropic->OpenAI) -> interception proxy (captures token_ids + logprobs) -> vLLM.
Claude Code owns its own tool loop; TRL reads the proxy trace, scores the workspace with a held-out verifier,
and trains with GRPO.

Two vLLM URLs (same as opencode):
  - `--vllm-url` (default localhost:8000): TRAINER <-> vLLM, NCCL weight-sync, stays local.
  - `--sandbox-vllm-url`: a public url the remote sandboxes reach vLLM through (a tunnel to your local one).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from typing import Any

from claude_code_env.config import ClaudeCodeConfig
from claude_code_env.harness import ClaudeCodeSessionFactory
from claude_code_env.task import ClaudeCodeTask
from datasets import Dataset, load_dataset
from openenv.core.harness import ResourceSession, ResourceSessionFactory, VerifyResult
from openenv.core.sandbox import HFSandboxBackend, SandboxHandle
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import (
    HarnessRolloutOutcome,
    HarnessRolloutWorker,
    has_tool_call,
)


# No pre-baked Claude Code image yet, so cold-install per rollout on a plain base. The HF sandbox execs as root,
# so every harness path hangs off /root.
SANDBOX_IMAGE = "python:3.12"
SANDBOX_HOME = "/root"
WORKDIR = f"{SANDBOX_HOME}/workdir"
# Claude Code's built-in tool names; restrict to the coding-relevant ones (no web / sub-agents).
ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]

# ============================================================================================================
# Dataset + held-out verifier (identical to opencode_hf_sandbox.py)
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

    def __call__(self, sandbox: SandboxHandle, task: ClaudeCodeTask) -> VerifyResult:
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
# Claude Code session factory (remote HF sandbox + in-sandbox shim + proxy)
# ============================================================================================================


class ClaudeCodeTaskFactory(ResourceSessionFactory):
    """Adapt the worker's `create(prompt=messages, seed, episode_id)` onto `ClaudeCodeSessionFactory`."""

    def __init__(self, inner: ClaudeCodeSessionFactory):
        self._inner = inner

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> ResourceSession:
        instruction = task[-1]["content"] if isinstance(task, list) and task else str(task)
        return self._inner.create(instruction, seed=seed, episode_id=episode_id)


def build_factory(
    sandbox_vllm_url: str, model: str, tests_by_id: dict, image: str, flavor: str, max_turns: int
) -> ClaudeCodeTaskFactory:
    config = ClaudeCodeConfig(
        base_url=sandbox_vllm_url,  # the in-sandbox proxy forwards here (public url / tunnel)
        model=model,  # proxy --model-override forces this exact id upstream
        sandbox_home=SANDBOX_HOME,
        agent_timeout_s=600.0,
        tools=ALLOWED_TOOLS,
        max_turns=max_turns,
    )
    inner = ClaudeCodeSessionFactory(
        config=config,
        sandbox_backend=HFSandboxBackend(image=image, flavor=flavor),
        mode="transparent_proxy",  # shim (Anthropic->OpenAI) + proxy capture token_ids + per_token_logps
        verifier=DeepCoderStdinVerifier(tests_by_id),
    )
    return ClaudeCodeTaskFactory(inner)


# ============================================================================================================
# Reward (application-owned hook)
# ============================================================================================================


def claude_code_reward(outcome: HarnessRolloutOutcome) -> float | None:
    """Binary terminal verifier + degeneracy penalties (mirror of opencode_reward, Claude Code tool names).

    - unscorable rollout -> None
    - never ran its code (no `Bash`) -> -0.1
    - else BINARY: all held-out tests pass -> 1.0; else 0.0
    - minus a step penalty for tool calls beyond a budget, capped at 0.5
    """
    step_budget, step_penalty, step_penalty_cap = 20, 0.03, 0.5
    frac = outcome.env_reward
    if frac is None:
        return None
    if outcome.tool_calls_by_name.get("Bash", 0) == 0:
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
    p.add_argument("--sandbox-vllm-url", required=True)
    p.add_argument("--sandbox-image", default=SANDBOX_IMAGE)
    p.add_argument("--sandbox-flavor", default="cpu-basic")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=16384)
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="async_grpo_claude_code_hf_sandbox")
    p.add_argument("--project", default="claude-code-hf-sandbox")
    p.add_argument("--trackio-space-id", default=None)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default=None)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()

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
        # Off: the completions printer can't render Claude Code's multi-turn trace
        # (the "completion" is a list of messages, not a string) and crashes the step.
        log_completions=False,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=build_factory(
            args.sandbox_vllm_url, args.model, tests_by_id, args.sandbox_image, args.sandbox_flavor, args.max_turns
        ),
        harness_adapter=None,  # loop-owning: Claude Code runs its own loop; TRL reads the proxy trace
        rollout_reward_fn=claude_code_reward,
        train_turn_fn=has_tool_call,  # reinforce only action turns, not prose
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],
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
