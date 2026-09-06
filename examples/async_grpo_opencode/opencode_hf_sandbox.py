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
#     "huggingface_hub>=1.22",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#     "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env",
# ]
# ///

"""AsyncGRPO training of the real `opencode` coding agent (loop-owning) in REMOTE Hugging Face sandboxes.

Same training path as `async_grpo_opencode.py`, but each rollout runs opencode in its own remote Hugging Face sandbox
(`HFSandboxBackend`) instead of a local subprocess, so rollouts scale out beyond a single node. opencode owns its
own tool loop; an in-sandbox proxy (`transparent_proxy` mode) forwards its calls to your vLLM server and captures
per-turn `(token_ids, logprobs)`. TRL reads that proxy trace, rebuilds training rows, scores the workspace with a
held-out verifier, and trains with GRPO.

Two vLLM URLs, on purpose:
  - `--vllm-url` (default `http://localhost:8000`): the TRAINER <-> vLLM link. Stays local for NCCL weight-sync.
  - `--sandbox-vllm-url`: a url the remote sandboxes use to reach that same vLLM (the in-sandbox proxy forwards
    there). Remote sandboxes cannot see `localhost`, so this must be reachable from outside: a public vLLM
    endpoint, or a tunnel to your local one (see below). Not tied to any tunnel provider.

Where opencode lives: nothing is installed per rollout. The default sandbox image
`ghcr.io/huggingface/openenv-opencode-sandbox:latest` pre-bakes the opencode CLI + the proxy (deps and
`interception.py`) under `/root`, so the harness skips the cold install. Pass `--sandbox-image python:3.12` to fall
back to cold-installing opencode + proxy deps per rollout.

Task: competitive-coding problems from `agentica-org/DeepCoder-Preview-Dataset`. The agent writes `solution.py`
(reads stdin, prints stdout); the verifier runs it against the problem's HELD-OUT tests (never shown to the agent)
and returns a DENSE reward = fraction passed. `opencode_reward` keeps that dense signal and adds small degeneracy
penalties. This whole file is self-contained and every training-facing object is module-level (picklable), so the
rollout worker can pickle the factory + verifier into its spawned child process.

Requirements:
  - An OpenAI-compatible vLLM server (see below), reachable locally by the trainer and publicly by the sandboxes.
  - An HF token with Jobs + Sandbox access in the environment (`HF_TOKEN`); each rollout is one HF sandbox.

Run (2 GPUs: vLLM on one, trainer on the other; a tunnel exposes vLLM to the sandboxes):

```sh
# Terminal 1 - serve the policy. Tool-calling + token-ids + NCCL weight-sync are all required.
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --max-model-len 98304 \
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2 - expose that vLLM publicly for the remote sandboxes.
cloudflared tunnel --no-autoupdate --url http://localhost:8000   # prints https://<name>.trycloudflare.com

# Terminal 3 - train. Trainer talks localhost (NCCL); sandboxes reach vLLM through the tunnel.
CUDA_VISIBLE_DEVICES=1 python examples/async_grpo_opencode/opencode_hf_sandbox.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --vllm-url http://localhost:8000 \
    --sandbox-vllm-url https://<name>.trycloudflare.com
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from typing import Any

from datasets import Dataset, load_dataset
from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.sandbox import HFSandboxBackend, SandboxHandle
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


# The HF sandbox image bakes opencode + the proxy under `/root`, and the sandbox execs as root, so `$HOME` and every
# harness path (`workdir`, `.opencode/bin`, `proxy/`) hang off `/root`. This is the only path difference from the
# local `async_grpo_opencode.py`, whose subprocess sandbox uses `/home/user`.
SANDBOX_IMAGE = "ghcr.io/huggingface/openenv-opencode-sandbox:latest"
SANDBOX_HOME = "/root"
WORKDIR = f"{SANDBOX_HOME}/workdir"


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
# opencode session factory (remote HF sandbox + in-sandbox proxy)
# ============================================================================================================


class OpencodeTaskFactory(ResourceSessionFactory):
    """Adapts the worker's `create(prompt=messages, seed, episode_id)` onto `OpenCodeSessionFactory`, which wants an
    `OpenCodeTask`: pull the instruction out of the last user message."""

    def __init__(self, inner: OpenCodeSessionFactory):
        self._inner = inner

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> ResourceSession:
        instruction = task[-1]["content"] if isinstance(task, list) and task else str(task)
        return self._inner.create(instruction, seed=seed, episode_id=episode_id)


def build_factory(
    sandbox_vllm_url: str, model: str, tests_by_id: dict, image: str, flavor: str
) -> OpencodeTaskFactory:
    config = OpenCodeConfig(
        provider="openai_compatible",
        base_url=f"{sandbox_vllm_url}/v1",  # the in-sandbox proxy forwards here; remote, so a public url (tunnel)
        model=model,  # proxy --model-override forces this exact id on upstream requests
        sandbox_home=SANDBOX_HOME,  # the HF sandbox execs as root; opencode + proxy are baked under /root
        agent_timeout_s=600.0,  # remote hop adds latency vs the local backend; give the edit/bash loop more room
        disabled_tools=["webfetch", "question", "task"],  # no web, no user, no sub-agents
        run_format="json",
        proxy_max_tokens_cap=8192,  # keep each turn's completion + the growing multi-turn prompt under --max-model-len
    )
    inner = OpenCodeSessionFactory(
        config=config,
        sandbox_backend=HFSandboxBackend(image=image, flavor=flavor),  # each rollout = its own remote HF sandbox
        mode="transparent_proxy",  # in-sandbox proxy captures completion_token_ids + per_token_logps
        verifier=DeepCoderStdinVerifier(tests_by_id),
    )
    return OpencodeTaskFactory(inner)


# ============================================================================================================
# Reward + turn-selection policy (application-owned; passed to the worker as hooks)
# ============================================================================================================


def opencode_reward(outcome: HarnessRolloutOutcome) -> float | None:
    """Dense terminal verifier + degeneracy penalties. Long-horizon credit is carried by the terminal reward,
    propagated to every trained token through the group-relative advantage.

      - unscorable rollout (no verifier score) -> None (dropped from the group baseline)
      - never ran its code (no `bash`) -> -0.1 (kills blind-write / prose-dump / give-up)
      - else DENSE base: the fraction of held-out tests passed (partial credit); timed out -> 0.0
      - minus a step penalty for tool calls beyond a budget (bounds runaway edit/bash loops), capped at 0.5
    """
    step_budget, step_penalty, step_penalty_cap = 30, 0.03, 0.5
    frac = outcome.env_reward
    bash = outcome.tool_calls_by_name.get("bash", 0)
    if frac is None:
        return None
    if bash == 0:
        return -0.1
    base = 0.0 if outcome.timed_out else frac  # dense: fraction of held-out tests passed (partial credit)
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
    p.add_argument("--vllm-url", default="http://localhost:8000")  # trainer <-> vLLM (weight sync, NCCL): local
    p.add_argument("--sandbox-vllm-url", required=True)  # public url the remote sandboxes reach vLLM through (tunnel)
    p.add_argument(
        "--sandbox-image", default=SANDBOX_IMAGE
    )  # pre-baked opencode+proxy; use python:3.12 to cold-install
    p.add_argument("--sandbox-flavor", default="cpu-basic")  # the agent only runs python/bash, no GPU needed
    p.add_argument(
        "--num-generations", type=int, default=8
    )  # >1 gives within-group pass/fail split -> nonzero advantage
    p.add_argument("--max-inflight", type=int, default=8)  # concurrent rollouts (each its own remote sandbox)
    p.add_argument("--max-completion-length", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-staleness", type=int, default=4)  # lower -> fresher rollouts -> ratios near 1 (more stable)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="async_grpo_opencode_hf_sandbox")
    p.add_argument("--project", default="opencode-hf-sandbox")
    p.add_argument("--trackio-space-id", default=None)  # optional: host the trackio dashboard on a HF Space
    p.add_argument("--push-to-hub", action="store_true")  # push the trained policy to the Hub at --hub-model-id
    p.add_argument("--hub-model-id", default=None)
    p.add_argument("--optim", default="adamw_torch")  # e.g. paged_adamw_8bit to fit a larger policy on one GPU
    p.add_argument("--gradient-checkpointing", action="store_true")  # trade compute for memory on a larger policy
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)  # more prompts per step -> smoother reward
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows, tests_by_id = build_dataset(n_prompts=args.n_prompts, seed=args.seed)
    dataset = Dataset.from_list(rows)

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        save_strategy="no",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    worker = HarnessRolloutWorker(
        harness_session_factory=build_factory(
            args.sandbox_vllm_url, args.model, tests_by_id, args.sandbox_image, args.sandbox_flavor
        ),
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
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
