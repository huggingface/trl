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

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from trl.environments import SandboxEnvironment

from .testing_utils import TrlTestCase


def _fake_sandbox(stdout="", stderr="", exit_code=0):
    """A stand-in for `huggingface_hub.Sandbox` with a `run` returning a canned result and a spy `kill`."""
    sandbox = MagicMock()
    sandbox.run.return_value = SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)
    return sandbox


class TestSandboxEnvironment(TrlTestCase):
    def test_exposes_run_as_only_tool(self):
        # The trainer probes the instance for public methods (excluding `reset`) to build the tool schema.
        env = SandboxEnvironment()
        tools = [
            name
            for name, _ in inspect.getmembers(env, predicate=inspect.ismethod)
            if not name.startswith("_") and name != "reset"
        ]
        assert tools == ["run"]

    def test_reset_boots_sandbox_once_and_reuses(self):
        env = SandboxEnvironment(image="alpine:3.20", flavor="cpu-basic")
        with patch("huggingface_hub.Sandbox") as sandbox_cls:
            sandbox_cls.create.return_value = _fake_sandbox()
            assert env.reset() is None
            env.run("echo hi")  # forces the background boot to resolve
            env.reset()  # instances are reused across rollouts: no new sandbox
            env.run("echo hi")
        sandbox_cls.create.assert_called_once_with(image="alpine:3.20", flavor="cpu-basic", idle_timeout="10m")

    def test_reset_does_not_block_on_boot(self):
        # `reset` must return without waiting for the (slow) VM boot, so it can't stall the trainer's rollout loop.
        env = SandboxEnvironment()
        with patch("huggingface_hub.Sandbox") as sandbox_cls:
            sandbox_cls.create.side_effect = lambda **kwargs: _fake_sandbox()
            env.reset()
            assert env._sandbox is None  # not yet resolved; the boot runs on the pool
            env.run("echo hi")  # resolves it on first use
            assert env._sandbox is not None

    def test_run_returns_combined_output(self):
        env = SandboxEnvironment()
        env._sandbox = _fake_sandbox(stdout="hello\n", stderr="warn\n")
        assert env.run("echo hello") == "hello\nwarn\n"
        env._sandbox.run.assert_called_once_with("echo hello", timeout=60, check=False)

    def test_run_reports_empty_output(self):
        env = SandboxEnvironment()
        env._sandbox = _fake_sandbox(stdout="", stderr="", exit_code=3)
        assert env.run("true") == "(no output, exit code 3)"

    def test_run_truncates_long_output(self):
        env = SandboxEnvironment()
        env._sandbox = _fake_sandbox(stdout="x" * 20000)
        output = env.run("cat big_file")
        assert output.endswith("... [truncated]")
        assert len(output) < 20000

    def test_del_kills_sandbox(self):
        env = SandboxEnvironment()
        sandbox = _fake_sandbox()
        env._sandbox = sandbox
        env.__del__()
        sandbox.kill.assert_called_once()
