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

import os
import subprocess
from pathlib import Path

import pytest

from ..testing_utils import TrlTestCase, require_torch_multi_accelerator


ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, env=env, cwd=ROOT)
    assert result.returncode == 0


@pytest.fixture
def get_config_path(lazy_shared_datadir):
    def _get_config_path(config_name):
        return lazy_shared_datadir / "accelerate_configs" / f"{config_name}.yaml"

    return _get_config_path


@require_torch_multi_accelerator
class TestDistributed(TrlTestCase):
    @pytest.mark.parametrize("config", ["ddp", "zero2", "zero3", "fsdp2"])
    def test_sft(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            "zero2",
            "zero3",
            pytest.param("fsdp2", marks=pytest.mark.xfail(reason="FSDP2 DPO is currently failing, see see #4812")),
        ],
    )
    def test_dpo(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/dpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize("config", ["ddp", "zero2", "zero3", "fsdp2"])
    def test_sft_dataset_streaming(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--dataset_streaming",
                "--max_steps", "3",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param("zero2", marks=pytest.mark.xfail(reason="ZeRO 2 is currently failing; see #4884")),
            pytest.param("zero3", marks=pytest.mark.xfail(reason="ZeRO 3 is currently failing; see #4831")),
            "fsdp2",
        ],
    )
    def test_sft_peft(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--use_peft",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize("config", ["ddp", "zero2", "zero3", "fsdp2"])
    def test_reward(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/reward.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_implicit_prompt_preference",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            "zero2",
            "zero3",
            pytest.param("fsdp2", marks=pytest.mark.xfail(reason="FSDP2 RLOO is currently failing, see #4854")),
        ],
    )
    def test_rloo(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/rloo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_prompt_only",
                "--reward_model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize("config", ["ddp", "zero2", "zero3", "fsdp2"])
    def test_grpo(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/grpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_prompt_only",
                "--reward_model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            ],
            os.environ.copy(),
        )
        # fmt: on
