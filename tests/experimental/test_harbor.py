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

"""Tests for the Harbor x TRL integration that don't need a running Harbor sandbox.

`harbor` is imported lazily (only when an env is *started*), so spec construction, agent resolution,
dataset building, and the reward function are all testable without `harbor` / a sandbox backend.
"""

from pathlib import Path

import pytest

from trl.experimental.harbor import AGENTS, HarborBashEnv, HarborEnv, HarborSpec
from trl.experimental.harbor._spec import _outcome_reward_func, _resolve_agent

from ..testing_utils import TrlTestCase


def _write_task(tasks_dir: Path, task_id: str, gold: str, difficulty: int) -> None:
    d = tasks_dir / task_id
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "instruction.md").write_text(f"Solve task {task_id}.")
    # Built from a joined list (not a triple-quoted block) so doc-builder doesn't reflow the TOML.
    lines = [
        "[task]",
        f'name = "{task_id}"',
        "[metadata]",
        f'gold_answer = "{gold}"',
        'reward_mode_initial = "exact_short"',
        f"difficulty_level = {difficulty}",
        f'kaggle_dataset_name = "owner/{task_id}"',
    ]
    (d / "task.toml").write_text("\n".join(lines))


class TestResolveAgent(TrlTestCase):
    def test_builtin_name(self):
        assert _resolve_agent("bash") is HarborBashEnv
        assert AGENTS["bash"] is HarborBashEnv

    def test_class_passthrough(self):
        assert _resolve_agent(HarborBashEnv) is HarborBashEnv

    def test_import_path(self):
        assert _resolve_agent("trl.experimental.harbor:HarborBashEnv") is HarborBashEnv

    def test_file_path(self):
        path = Path(self.tmp_dir) / "my_harness.py"
        path.write_text(
            "from trl.experimental.harbor import HarborEnv\n"
            "class MyEnv(HarborEnv):\n"
            "    def run_cmd(self, command: str) -> str:\n"
            "        'Run a command.\\n\\nArgs:\\n    command: cmd.'\n"
            "        return self._exec(command)\n"
        )
        cls = _resolve_agent(f"{path}:MyEnv")
        assert issubclass(cls, HarborEnv) and cls.__name__ == "MyEnv"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            _resolve_agent("not-a-harness")

    def test_non_harborenv_raises(self):
        with pytest.raises(TypeError):
            _resolve_agent("trl.experimental.harbor:HarborSpec")  # not a HarborEnv subclass


class TestHarborSpecDataset(TrlTestCase):
    def _suite(self) -> str:
        tasks = Path(self.tmp_dir) / "tasks"
        tasks.mkdir()
        _write_task(tasks, "0001_a", "alpha", 0)
        _write_task(tasks, "0002_b", "beta", 3)
        return str(self.tmp_dir)

    def test_train_dataset_columns_and_metadata(self):
        ds = HarborSpec(self._suite()).train_dataset
        assert len(ds) == 2
        assert ds[0]["prompt"] == [{"role": "user", "content": ""}]  # env appends instruction at reset
        assert ds[0]["task_dir"].endswith("0001_a")
        assert ds[0]["task_index"] == 0
        assert ds[0]["gold_answer"] == "alpha"
        assert ds[1]["difficulty_level"] == 3

    def test_num_tasks_cap(self):
        ds = HarborSpec(self._suite(), num_tasks=1).train_dataset
        assert len(ds) == 1

    def test_indices_selection(self):
        ds = HarborSpec(self._suite(), indices=[1]).train_dataset
        assert len(ds) == 1 and ds[0]["task_dir"].endswith("0002_b")

    def test_num_tasks_and_indices_mutually_exclusive(self):
        with pytest.raises(ValueError):
            HarborSpec(self._suite(), num_tasks=1, indices=[0])

    def test_environment_factory_returns_fresh_envs(self):
        factory = HarborSpec(self._suite(), agent="bash").environment_factory
        e1, e2 = factory(), factory()
        assert isinstance(e1, HarborBashEnv) and e1 is not e2


class TestRewardFunc(TrlTestCase):
    def test_outcome_reward_reads_env_reward(self):
        class _Env:
            def __init__(self, r):
                self.reward = r

        assert _outcome_reward_func([_Env(1.0), _Env(0.0)]) == [1.0, 0.0]

    def test_outcome_reward_uses_environment_reward_when_passed(self):
        # AsyncGRPOTrainer captures rewards in its rollout worker and passes them as a list, with no
        # live env instances. The reward func must use them directly.
        assert _outcome_reward_func(environment_reward=[0.25, 0.75]) == [0.25, 0.75]
