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

from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass

from ...scripts.utils import TrlParser


@dataclass(slots=True)
class CommandContext:
    """Context shared by CLI commands during execution."""

    parser: TrlParser
    argv: list[str]
    launch_args: list[str]

    def argv_after(self, token: str) -> list[str]:
        """
        Return CLI tokens after the first occurrence of `token`.

        Parameters:
            token (`str`):
                Subcommand name as it appears in `argv`.
        """
        try:
            index = self.argv.index(token)
        except ValueError:
            return []
        return self.argv[index + 1 :]


class Command(ABC):
    """
    Base command definition for the TRL CLI.

    Parameters:
        name (`str`):
            Subcommand name exposed by the CLI.
        help_text (`str`):
            Short description displayed in help output.
        uses_accelerate (`bool`, *optional*, defaults to `False`):
            Whether this command uses `accelerate launch`.
    """

    def __init__(self, name: str, help_text: str, uses_accelerate: bool = False):
        self.name = name
        self.help_text = help_text
        self.uses_accelerate = uses_accelerate

    @abstractmethod
    def register(self, subparsers) -> None:
        """Register this command parser in the subparser collection."""

    @abstractmethod
    def run(self, args: Namespace, context: CommandContext) -> int:
        """Execute the command."""
