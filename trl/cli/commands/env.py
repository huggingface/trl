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

from ...scripts.env import print_env
from .base import Command, CommandContext


class EnvCommand(Command):
    """CLI command that prints TRL environment information."""

    def __init__(self):
        super().__init__(name="env", help_text="Print the environment information")

    def register(self, subparsers) -> None:
        subparsers.add_parser(self.name, help=self.help_text)

    def run(self, args: Namespace, context: CommandContext) -> int:
        print_env()
        return 0
