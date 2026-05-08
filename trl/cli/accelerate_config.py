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

import importlib.resources as resources
from pathlib import Path


def resolve_accelerate_config_argument(launch_args: list[str]) -> list[str]:
    """
    Resolve `--accelerate_config` from CLI arguments into `accelerate --config_file`.

    The function supports either a filesystem path or a predefined config name shipped in `trl/accelerate_configs`
    (without the `.yaml` suffix).
    """
    if "--accelerate_config" not in launch_args:
        return launch_args

    config_index = launch_args.index("--accelerate_config")
    if config_index + 1 >= len(launch_args):
        raise ValueError("Expected a value after `--accelerate_config`.")

    config_name = launch_args[config_index + 1]
    if Path(config_name).is_file():
        accelerate_config_path = config_name
    else:
        candidate = resources.files("trl.accelerate_configs").joinpath(f"{config_name}.yaml")
        if not candidate.exists():
            raise ValueError(
                f"Accelerate config {config_name} is neither a file nor a valid config in the `trl` package. "
                "Please provide a valid config name or a path to a config file."
            )
        accelerate_config_path = candidate

    # Remove '--accelerate_config <value>'.
    launch_args = launch_args[:config_index] + launch_args[config_index + 2 :]
    return ["--config_file", str(accelerate_config_path)] + launch_args
