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

from ...scripts.vllm_serve import main as vllm_serve_main
from ...scripts.vllm_serve import make_parser as make_vllm_serve_parser
from .base import Command, CommandContext


class VllmServeCommand(Command):
    """CLI command for serving TRL models with vLLM."""

    def __init__(self):
        super().__init__(name="vllm-serve", help_text="Serve a model with vLLM")

    def register(self, subparsers) -> None:
        make_vllm_serve_parser(subparsers)

    def run(self, args: Namespace, context: CommandContext) -> int:
        (script_args,) = context.parser.parse_args_and_config(args=context.argv)
        vllm_serve_main(script_args)
        return 0
