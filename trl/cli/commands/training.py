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

import importlib
import importlib.resources as resources
import os
import subprocess
import sys
import warnings
from argparse import Namespace

from .base import Command, CommandContext


def _subtract_subsequence(lst: list[str], subseq: list[str]) -> list[str]:
    """Return lst with the ordered subsequence subseq removed."""
    sub_iter = iter(subseq)
    current = next(sub_iter, None)
    result = []
    for item in lst:
        if current is not None and item == current:
            current = next(sub_iter, None)
        else:
            result.append(item)
    return result


class TrainingCommand(Command):
    """
    Generic CLI command that launches a training script with torchrun.

    The script `trl/scripts/<name>.py` must expose a `make_parser()` function.

    Parameters:
        name (`str`):
            CLI subcommand name (e.g. `"dpo"`).
    """

    def __init__(self, name: str):
        super().__init__(name=name, help_text=f"Run the {name} training script")

    def register(self, subparsers) -> None:
        subparsers.add_parser(self.name, help=self.help_text, add_help=False)

    def run(self, args: Namespace, context: CommandContext) -> int:
        import torch
        from accelerate.commands.config import default_config_file
        from torch.distributed.run import get_args_parser, parse_args, run

        from ..accelerate_config import resolve_accelerate_config_argument
        from ..accelerate_launcher import launch_training_script

        module = importlib.import_module(f"...scripts.{self.name}", package=__package__)
        all_args = context.argv_after(self.name)
        parser = module.make_parser(prog=f"trl {self.name}")

        # Handles -h (exits). Returns config_remaining and cli_remaining separately.
        # cli_remaining is an ordered subsequence of all_args; config_remaining is not.
        *_, config_remaining, cli_remaining = parser.parse_args_and_config(
            all_args, return_remaining_strings=True, separate_remaining_strings=True
        )
        # Arguments the training script does not know are the launcher's (e.g. `--nproc_per_node 4`).
        launch_args = config_remaining + cli_remaining
        training_script_args = _subtract_subsequence(all_args, cli_remaining)
        training_script = str(resources.files("trl.scripts").joinpath(f"{self.name}.py"))

        # Arguments torchrun does not know (`accelerate launch` ones, like `--num_processes` or `--accelerate_config`)
        # and an `accelerate config` default file still launch through accelerate, until v2.0.0
        _, unknown_args = get_args_parser().parse_known_args([*launch_args, training_script])
        if unknown_args or (not launch_args and os.path.isfile(default_config_file)):
            if unknown_args:
                reason = f"of arguments torchrun does not know: {' '.join(unknown_args)}"
            else:
                reason = f"of the `accelerate config` default file {default_config_file}"
            warnings.warn(
                f"`trl {self.name}` launched with `accelerate launch` because {reason}. Launching with `accelerate "
                "launch` is deprecated and will be removed in v2.0.0: pass torchrun arguments instead (such as "
                "`--nproc_per_node`), and configure DeepSpeed or FSDP with `--deepspeed` or `--fsdp`.",
                FutureWarning,
                stacklevel=2,
            )
            launch_training_script(
                script_name=f"{self.name}.py",
                launch_args=resolve_accelerate_config_argument(launch_args),
                training_script_args=training_script_args,
            )
            return 0

        nproc_per_node = max(torch.accelerator.device_count(), 1)
        if not launch_args and nproc_per_node == 1:
            # A single process needs no rendezvous: run the script as `python script.py` would.
            subprocess.run([sys.executable, training_script, *training_script_args], check=True)
            return 0
        if "--nproc_per_node" not in launch_args:
            launch_args = ["--nproc_per_node", str(nproc_per_node), *launch_args]
        run(parse_args([*launch_args, training_script, *training_script_args]))
        return 0
