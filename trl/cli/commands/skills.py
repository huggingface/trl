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

from argparse import Namespace

from ...skills.cli import add_skills_subcommands
from .base import Command, CommandContext


class SkillsCommand(Command):
    """CLI command that manages TRL agent skills."""

    def __init__(self):
        super().__init__(name="skills", help_text="Manage TRL agent skills")
        self._skills_parser = None

    def register(self, subparsers) -> None:
        self._skills_parser = subparsers.add_parser(self.name, help=self.help_text)
        skills_subparsers = self._skills_parser.add_subparsers(dest="skills_command", help="Skills commands")
        add_skills_subcommands(skills_subparsers)

    def run(self, args: Namespace, context: CommandContext) -> int:
        if getattr(args, "skills_command", None):
            if hasattr(args, "func"):
                return args.func(args)
            print("Error: Unknown skills command")
            return 1

        if self._skills_parser is not None:
            self._skills_parser.print_help()
        return 0
