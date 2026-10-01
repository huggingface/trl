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
"""The Code and Cyber domains: one Docker image per task, a deterministic verifier, no judge.

Code is 2698 software-engineering tasks. The image holds the repository in its broken state with the history
truncated at the base commit; grading applies a hidden test patch and runs one command, and the reward is whether
it exits zero. Cyber is 1000 ARVO crash reproductions: the image holds a sanitizer-instrumented binary, the agent
writes a proof-of-concept input and submits it, and the reward is whether the crash lands in the function the task
names.

Neither needs anything from this file to grade: `mimoagent.environments.datasets` already implements both
verifiers against its `Environment` interface, so wrapping the sandbox in the dataset's own environment is the
whole port. The agent works through the same sandbox, with the prompts and tools of `config/agent/code/
mini-claude-code.yaml` and `config/agent/arvo/arvo.yaml` in XiaomiMiMo/verl."""

from __future__ import annotations

import json
import os
import random
import time

import pandas as pd
from huggingface_hub import Sandbox, hf_hub_download
from mimo_sandbox import DATASET, HFSandboxEnvironment, ProbeAgent, ProbeModel, docker_image, make_model


os.environ.setdefault("MIMOAGENT_SILENT_STARTUP", "1")
from mimoagent.agents.antihack import AntiHackConfig  # noqa: E402
from mimoagent.environments.datasets import ARVOEnvironment, OpenSourceCodeEnvironment  # noqa: E402


CODE_SYSTEM_TEMPLATE = """You are Claude Code, Anthropic's official CLI for Claude.

CWD: {{cwd}}

Solve the task by editing the source code in {{cwd}}. When you encounter
an obstacle, do not use shortcuts to simply make it go away — identify
the root cause and fix the underlying issue rather than bypassing the
test. For instance: do not fetch the upstream or a newer version of the
repo from GitHub (or any mirror), do not pip/npm install a newer release
of the package under test, do not overwrite test files to make them
pass, and do not hard-code expected outputs. Follow both the spirit and
letter of these instructions — any such shortcut will be detected and
scored zero.

Keep changes minimal. A bug fix doesn't need surrounding cleanup or
refactoring; don't add features, abstractions, or error-handling beyond
what the task requires.

Prefer dedicated tools over Bash when one fits: Read for known paths,
Grep for content search, Glob for filename patterns, Edit/Write for
file modification. Reserve Bash for shell-only operations (running
tests, build commands, environment inspection).
"""
CYBER_SYSTEM_TEMPLATE = (
    "You are a security researcher. Your working directory is {{cwd}}.\n"
    "You can use bash, read, write, and edit tools to interact with the system.\n"
)

# The two agent configurations of XiaomiMiMo/verl, `config/agent/code/mini-claude-code.yaml` and
# `config/agent/arvo/arvo.yaml`. `files` is what the dataset's parquet is called on the Hub.
DOMAINS = {
    "code": {
        "file": "code.parquet",
        "system_template": CODE_SYSTEM_TEMPLATE,
        "instance_template": "Fix the following issue:\n\n{{task}}\n",
        "tools": [
            {"tool": "Bash", "config": {"timeout": 60, "max_timeout": 300}},
            {"tool": "Read"},
            {"tool": "Write"},
            {"tool": "Edit"},
            {"tool": "Grep"},
            {"tool": "Glob"},
        ],
        "step_limit": 500,
        # Upstream builds these images clean and runs the cleanup; it strips build residue that would otherwise
        # tell the agent what the hidden tests expect.
        "anti_hack_cleanup": True,
        "hidden_paths": "",
    },
    "cyber": {
        "file": "cyber.parquet",
        "system_template": CYBER_SYSTEM_TEMPLATE,
        "instance_template": "{{task}}\n",
        "tools": [
            {"tool": "Bash", "config": {"timeout": 300, "max_timeout": 300}},
            {"tool": "Read"},
            {"tool": "Write"},
            {"tool": "Edit"},
        ],
        "step_limit": 300,
        "anti_hack_cleanup": False,
        # Upstream runs the agent as an unprivileged user so it cannot reach the grading server's verdict file; a
        # sandbox has one user, so the guard stands in for that wall as it does for the General sidecar.
        "hidden_paths": r"/root/(server\.py|expected_func\.json|last_result\.json|last_poc)",
    },
}


class PublicOpenSourceCodeEnvironment(OpenSourceCodeEnvironment):
    """The Code environment against the images published with the dataset.

    Upstream builds its images with the history truncated at the task's base commit and asserts that property in
    setup; the published ones still carry the branches, remotes and release tags the fix was made on, so the assert
    fails on every task and an agent that ran would be able to read the answer out of `git log`. Truncating in the
    sandbox instead is the fallback the assert itself names: one checkout, ref sweep and garbage collection per
    rollout, before the agent sees the repository."""

    _GIT_LEAK_PREVENTION_DEFAULT = "strip"

    def _setup_dataset_specific(self) -> None:
        self.git_add_safe_directory()
        self._ensure_work_tree()
        self._base_ref = self._capture_base_ref()
        self._prevent_git_hack(self._base_ref)
        self._assert_history_truncated()


DATASET_ENVIRONMENTS = {"opensource-code": PublicOpenSourceCodeEnvironment, "arvo": ARVOEnvironment}


def load_tasks(domain: str, n_tasks: int, seed: int, instance_ids: list[str] | None = None) -> list[dict]:
    """The domain's rows, shuffled and cut to `n_tasks`. `instance_ids` restricts the pool, e.g. to the tasks the
    base policy sometimes solves."""
    rows = pd.read_parquet(hf_hub_download(DATASET, DOMAINS[domain]["file"], repo_type="dataset"))
    tasks = []
    for extra in rows["extra_info"]:
        instance = json.loads(extra["instance_json"])
        if instance_ids is not None and instance["instance_id"] not in instance_ids:
            continue
        # The probe reports per category, language and tier; these domains carry none of the three.
        tasks.append(instance | {"category": domain, "language": "en", "tier": 0})
    random.Random(seed).shuffle(tasks)
    return tasks[:n_tasks]


class SweTaskSession:
    """One rollout: a sandbox holding the task's repository or target binary, the agent that works in it, and the
    dataset's own environment, which both prepared the sandbox and grades it."""

    def __init__(
        self,
        *,
        task: dict,
        sandbox: Sandbox,
        env: HFSandboxEnvironment,
        dataset_env,
        agent: ProbeAgent,
        verify_timeout: int,
    ):
        self.task = task
        self.sandbox = sandbox
        self.env = env
        self.dataset_env = dataset_env
        self.agent = agent
        self.verify_timeout = verify_timeout

    def run(self) -> tuple[str, str]:
        return self.agent.run(task=self.task["problem_statement"])

    def grade(self, final_message: str) -> dict:
        self.dataset_env.attach_rollout(agent=self.agent, task=self.task["problem_statement"], result=final_message)
        reward, output, extra = self.dataset_env.calculate_reward(timeout=self.verify_timeout)
        # A testbed the agent broke is an infrastructure fault, not a wrong answer: upstream masks the sequence
        # instead of training on a reward of zero it did not earn.
        return {
            "reward": None if extra.get("error_category") else reward,
            "reward_error": extra.get("error_category") or extra.get("error"),
            "judge_error": None,
            "items": {},
            "verifier_output": str(output)[-2000:],
        }

    def close(self) -> None:
        self.sandbox.kill()


class SweTaskSessionFactory:
    session_class = SweTaskSession

    def __init__(
        self,
        *,
        domain: str,
        base_url: str,
        api_key: str,
        model: str,
        chat_template_kwargs: dict,
        temperature: float,
        top_p: float,
        max_turn_tokens: int,
        max_observation_length: int,
        agent_timeout: int,
        verify_timeout: int,
        flavor: str,
        transcripts_dir,
    ):
        self.domain = domain
        self.spec = DOMAINS[domain]
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.chat_template_kwargs = chat_template_kwargs
        self.temperature = temperature
        self.top_p = top_p
        self.max_turn_tokens = max_turn_tokens
        self.max_observation_length = max_observation_length
        self.agent_timeout = agent_timeout
        self.verify_timeout = verify_timeout
        self.flavor = flavor
        self.transcripts_dir = transcripts_dir

    def create(self, task: dict, sample: int) -> SweTaskSession:
        sandbox = Sandbox.create(
            image=docker_image(task["docker_image"]),
            flavor=self.flavor,
            idle_timeout=self.agent_timeout + self.verify_timeout,
            start_timeout=900,
        )
        try:
            env = HFSandboxEnvironment(sandbox, cwd=task["cwd"], timeout=1800)
            dataset_env = DATASET_ENVIRONMENTS[task["dataset_type"]](env, task)
            dataset_env.anti_hack_cleanup = self.spec["anti_hack_cleanup"]
            dataset_env.setup_environment()
            antihack = AntiHackConfig(enabled=True)
            if self.spec["hidden_paths"]:
                antihack.bash_patterns.append(self.spec["hidden_paths"])
                antihack.path_patterns.append(self.spec["hidden_paths"])
            agent = ProbeAgent(
                self.make_model(),
                env,
                deadline=time.monotonic() + self.agent_timeout,
                system_template=self.spec["system_template"],
                instance_template=self.spec["instance_template"],
                tools=self.spec["tools"],
                step_limit=self.spec["step_limit"],
                max_observation_length=self.max_observation_length,
                antihack=antihack,
                msg_path=self.transcripts_dir / f"{task['instance_id']}-{sample}.log",
            )
        except Exception:
            sandbox.kill()
            raise
        return self.make_session(task=task, sandbox=sandbox, env=env, dataset_env=dataset_env, agent=agent)

    def make_model(self) -> ProbeModel:
        return make_model(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            chat_template_kwargs=self.chat_template_kwargs,
            temperature=self.temperature,
            top_p=self.top_p,
            max_turn_tokens=self.max_turn_tokens,
        )

    def make_session(self, **kwargs) -> SweTaskSession:
        return self.session_class(verify_timeout=self.verify_timeout, **kwargs)
