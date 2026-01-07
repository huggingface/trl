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
CONFIG_PATH = ROOT / "tests" / "accelerate_configs" / "2gpu.yaml"


def run_command(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, env=env, cwd=ROOT)
    assert result.returncode == 0


@require_torch_multi_accelerator
class TestDistributed(TrlTestCase):
    def test_sft(self):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
            ],
            os.environ.copy(),
        )
        # fmt: on

    def test_dpo(self):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/dpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
            ],
            os.environ.copy(),
        )
        # fmt: on

    def test_sft_streaming(self):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/sft.py",
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

    @pytest.mark.xfail(reason="PEFT + multi-GPU is broken, see https://github.com/huggingface/trl/issues/4782")
    def test_sft_peft(self):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--use_peft",
            ],
            os.environ.copy(),
        )
        # fmt: on
