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

from pathlib import Path
from unittest.mock import patch


def test_cli_accelerate_config_overrides_config_file(tmp_path):
    """A command-line Accelerate config must override the YAML config value."""
    from trl.cli import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text("accelerate_config: zero2\n")

    with patch("trl.cli.accelerate_launcher.launch_training_script") as mock_launch:
        main(["sft", "--config", str(config_path), "--accelerate_config", "zero3"])

    mock_launch.assert_called_once()
    launch_args = mock_launch.call_args.kwargs["launch_args"]
    assert Path(launch_args[1]).name == "zero3.yaml"
    assert "--accelerate_config" not in launch_args
    assert mock_launch.call_args.kwargs["training_script_args"] == ["--config", str(config_path)]
