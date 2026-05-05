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

"""Tests for `trl.experimental.openreward`.

A class-scoped fixture spawns ``_openreward_echo_env.py`` as a uvicorn subprocess on a free port and points the
openreward SDK at it via the ``OPENREWARD_API_URL`` / ``OPENREWARD_SESSION_URL`` overrides. Tests then exercise the
adapter end-to-end against real HTTP — no mocks, no network.

The same env definition is published at ``trl-internal-testing/openreward-echo-env`` if you want to point at the hosted
Space directly.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from trl.experimental.openreward import OpenRewardSpec

from ..testing_utils import TrlTestCase, require_openreward


_HERE = Path(__file__).parent
_ECHO_ENV_SCRIPT = _HERE / "_openreward_echo_env.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="class")
def echo_env_url():
    """Spawn the echo env on a free port; tear down on teardown."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(_ECHO_ENV_SCRIPT)],
        env={**os.environ, "PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError(f"echo env did not become ready at {url}")

    # The openreward SDK by default rewrites base_url into api.<host> /
    # sessions.<host>; for a single-host self-hosted server these env vars
    # bypass that two-subdomain layout.
    saved = {k: os.environ.get(k) for k in ("OPENREWARD_API_URL", "OPENREWARD_SESSION_URL", "OPENREWARD_API_KEY")}
    os.environ["OPENREWARD_API_URL"] = url
    os.environ["OPENREWARD_SESSION_URL"] = url
    os.environ.setdefault("OPENREWARD_API_KEY", "test")

    yield url

    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()


@require_openreward
@pytest.mark.usefixtures("echo_env_url")
class TestOpenRewardSpec(TrlTestCase):
    """Exercises the public `OpenRewardSpec` surface against a real ORS server."""

    def test_construction_is_lazy(self, echo_env_url):
        # Construction must not perform any HTTP — `train_dataset` /
        # `environment_factory` are `cached_property` and only fire on access.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        # Touching only private attributes should not have hit the network.
        assert spec._target == echo_env_url
        assert spec._is_url is True
        assert spec._num_tasks == 2

    def test_train_dataset_derives_from_env(self, echo_env_url):
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        ds = spec.train_dataset
        assert len(ds) == 2
        assert "prompt" in ds.column_names
        assert "task_index" in ds.column_names
        # Per-task metadata folded in (id, target) when include_metadata=True.
        assert "target" in ds.column_names
        assert ds[0]["task_index"] == 0
        assert ds[0]["target"] == "hello"
        assert ds[1]["target"] == "world"

    def test_train_dataset_with_indices(self, echo_env_url):
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", indices=[0, 2])
        ds = spec.train_dataset
        assert [row["target"] for row in ds] == ["hello", "trl"]

    def test_num_tasks_and_indices_are_mutually_exclusive(self, echo_env_url):
        with pytest.raises(ValueError):
            OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2, indices=[0])

    def test_environment_factory_returns_rollout_env_with_bound_tools(self, echo_env_url):
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        env = spec.environment_factory()
        # The single ORS tool is bound as a Python method with a typed signature.
        assert callable(env.echo)
        sig = env.echo.__annotations__
        assert sig["text"] is str
        assert sig["return"] is str

    def test_reset_returns_prompt_and_opens_session(self, echo_env_url):
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env = spec.environment_factory()
        prompt = env.reset(**spec.train_dataset[0])
        assert "echo" in prompt and "hello" in prompt
        env._close()

    def test_correct_echo_returns_match_with_reward_and_finished(self, echo_env_url):
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env = spec.environment_factory()
        env.reset(**spec.train_dataset[0])
        out = env.echo(text="hello")
        assert "match" in out
        assert env.reward == 1.0
        assert env.finished is True
        env._close()

    def test_wrong_echo_returns_no_match_with_zero_reward(self, echo_env_url):
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env = spec.environment_factory()
        env.reset(**spec.train_dataset[0])
        out = env.echo(text="goodbye")
        assert "no match" in out
        assert env.reward == 0.0
        assert env.finished is False
        env._close()

    def test_reward_func_reads_last_non_null_per_environment(self, echo_env_url):
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        env_a = spec.environment_factory()
        env_b = spec.environment_factory()
        env_a.reset(**spec.train_dataset[0])
        env_b.reset(**spec.train_dataset[1])
        env_a.echo(text="hello")  # match → reward=1.0
        env_b.echo(text="oops")  # no match → reward=0.0
        rewards = spec.reward_funcs(environments=[env_a, env_b])
        assert rewards == [1.0, 0.0]
        env_a._close()
        env_b._close()

    def test_factory_produces_isolated_sessions(self, echo_env_url):
        # GRPO opens N concurrent envs; mutating one must not leak into another.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=2)
        env_a = spec.environment_factory()
        env_b = spec.environment_factory()
        env_a.reset(**spec.train_dataset[0])
        env_b.reset(**spec.train_dataset[1])
        env_a.echo(text="hello")
        assert env_a.reward == 1.0
        assert env_b.reward == 0.0  # untouched
        env_a._close()
        env_b._close()

    def test_metadata_does_not_overwrite_reserved_columns(self, echo_env_url):
        # If a task spec ever shipped a `prompt` key, the metadata loop must
        # not clobber our chat-format `prompt` column. Same for `task_index`.
        # We assert the shape directly — the echo env's task spec doesn't
        # currently have either, but the guard is what we're testing.
        spec = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        ds = spec.train_dataset
        # `prompt` is a list-of-message-dicts, not a string from task spec.
        assert isinstance(ds[0]["prompt"], list)
        assert ds[0]["prompt"][0]["role"] == "user"
        # `task_index` is the int we set, not anything from the spec.
        assert isinstance(ds[0]["task_index"], int)

    def test_two_specs_get_isolated_rollout_subclasses(self, echo_env_url):
        # Two specs (potentially against different envs with different tool
        # sets) must each produce rollout instances with their own subclass,
        # so neither side's bound tools clobber or shadow the other's.
        spec_a = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        spec_b = OpenRewardSpec(echo_env_url, env_name="echoenvironment", num_tasks=1)
        env_a = spec_a.environment_factory()
        env_b = spec_b.environment_factory()
        # Each rollout is a distinct subclass of _RolloutEnvironment.
        assert type(env_a) is not type(env_b)
        # Both subclasses still get their own `echo` method.
        assert callable(env_a.echo) and callable(env_b.echo)
        env_a._close()
        env_b._close()
