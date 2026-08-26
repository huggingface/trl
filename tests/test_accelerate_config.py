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

import pytest

from trl.cli.accelerate_config import resolve_accelerate_config_argument


@pytest.mark.parametrize(
    "config_argument", [["--accelerate_config", "single_gpu"], ["--accelerate_config=single_gpu"]]
)
def test_resolve_packaged_accelerate_config(config_argument):
    launch_args = resolve_accelerate_config_argument([*config_argument, "--num_processes", "2"])

    assert launch_args[0] == "--config_file"
    assert Path(launch_args[1]).name == "single_gpu.yaml"
    assert launch_args[2:] == ["--num_processes", "2"]


@pytest.mark.parametrize("config_argument", ["--accelerate_config", "--accelerate_config="])
def test_resolve_accelerate_config_requires_value(config_argument):
    with pytest.raises(ValueError, match="Expected a value"):
        resolve_accelerate_config_argument([config_argument])
