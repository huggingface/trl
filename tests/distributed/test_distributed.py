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
import tempfile
from pathlib import Path

from ..testing_utils import require_torch_multi_accelerator


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "tests" / "accelerate_configs" / "2gpu.yaml"


def run_command(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, env=env, cwd=ROOT)
    assert result.returncode == 0


@require_torch_multi_accelerator
def test_sft():
    with tempfile.TemporaryDirectory() as tmpdir:
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/sft.py",
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--output_dir", tmpdir,
            ],
            os.environ.copy(),
        )
        # fmt: on


@require_torch_multi_accelerator
def test_dpo():
    with tempfile.TemporaryDirectory() as tmpdir:
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/dpo.py",
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
                "--output_dir", tmpdir,
            ],
            os.environ.copy(),
        )
        # fmt: on


@require_torch_multi_accelerator
def test_sft_streaming():
    with tempfile.TemporaryDirectory() as tmpdir:
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/sft.py",
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--dataset_streaming",
                "--max_steps", "3",
                "--output_dir", tmpdir,
            ],
            os.environ.copy(),
        )
        # fmt: on
