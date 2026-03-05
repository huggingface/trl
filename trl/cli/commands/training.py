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
    Generic CLI command that launches a training script with accelerate.

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
        from ..accelerate_config import resolve_accelerate_config_argument
        from ..accelerate_launcher import launch_training_script

        module = importlib.import_module(f"...scripts.{self.name}", package=__package__)
        all_args = context.argv_after(self.name)
        parser = module.make_parser()

        # Handles -h (exits). Returns config_remaining and cli_remaining separately.
        # cli_remaining is an ordered subsequence of all_args; config_remaining is not.
        *_, config_remaining, cli_remaining = parser.parse_args_and_config(
            all_args, return_remaining_strings=True, separate_remaining_strings=True
        )
        launch_args = resolve_accelerate_config_argument(config_remaining + cli_remaining)
        training_script_args = _subtract_subsequence(all_args, cli_remaining)

        launch_training_script(
            script_name=f"{self.name}.py",
            launch_args=launch_args,
            training_script_args=training_script_args,
        )
        return 0
