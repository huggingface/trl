# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import subprocess
import sys
import unittest
from pathlib import Path
import os
import glob
from trl.commands.cli_utils import populate_supported_commands


@unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")
def test_sft_cli():
    try:
        subprocess.run(
            "trl sft --max_steps 1 --output_dir tmp-sft --model_name_or_path trl-internal-testing/tiny-random-LlamaForCausalLM --dataset_name stanfordnlp/imdb --learning_rate 1e-4 --lr_scheduler_type cosine --dataset_text_field text",
            shell=True,
            check=True,
        )
    except BaseException as exc:
        raise AssertionError("An error occured while running the CLI, please double check") from exc


@unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")
def test_dpo_cli():
    try:
        subprocess.run(
            "trl dpo --max_steps 1 --output_dir tmp-dpo --model_name_or_path trl-internal-testing/tiny-random-LlamaForCausalLM --dataset_name trl-lib/ultrafeedback_binarized --learning_rate 1e-4 --lr_scheduler_type cosine",
            shell=True,
            check=True,
        )
    except BaseException as exc:
        raise AssertionError("An error occured while running the CLI, please double check") from exc


def test_env_cli():
    output = subprocess.run("trl env", capture_output=True, text=True, shell=True, check=True)
    assert "- Python version: " in output.stdout


def test_populate_supported_commands():
    commands = populate_supported_commands()
    
    # Check for specific commands
    assert 'sft' in commands, "SFT command not found"
    assert 'dpo' in commands, "DPO command not found"
    
    # Check that all commands are strings and don't have .py extension
    for cmd in commands:
        assert isinstance(cmd, str), f"Command {cmd} is not a string"
        assert not cmd.endswith('.py'), f"Command {cmd} should not have .py extension"
    
    # Check that the number of commands matches the number of .py files in the scripts directory
    trl_dir = Path(__file__).resolve().parent.parent
    scripts_path = os.path.join(trl_dir, 'examples', 'scripts', '*.py')
    py_files = glob.glob(scripts_path)
    assert len(commands) == len(py_files), f"Number of commands ({len(commands)}) doesn't match number of .py files ({len(py_files)})"