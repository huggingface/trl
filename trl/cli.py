# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import sys

from accelerate.commands.launch import launch_command, launch_command_parser

from .scripts.chat import main as chat_main
from .scripts.chat import make_parser as make_chat_parser
from .scripts.dpo import make_parser as make_dpo_parser
from .scripts.env import print_env
from .scripts.grpo import make_parser as make_grpo_parser
from .scripts.kto import make_parser as make_kto_parser
from .scripts.sft import make_parser as make_sft_parser
from .scripts.utils import TrlParser
from .scripts.vllm_serve import main as vllm_serve_main
from .scripts.vllm_serve import make_parser as make_vllm_serve_parser


def main():
    parser = TrlParser(prog="TRL CLI", usage="trl", allow_abbrev=False)

    # Add the subparsers
    subparsers = parser.add_subparsers(help="available commands", dest="command", parser_class=TrlParser)

    # Add the subparsers for every script
    make_chat_parser(subparsers)
    make_dpo_parser(subparsers)
    subparsers.add_parser("env", help="Print the environment information")
    make_grpo_parser(subparsers)
    make_kto_parser(subparsers)
    make_sft_parser(subparsers)
    make_vllm_serve_parser(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    if args.command == "chat":
        (chat_args,) = parser.parse_args_and_config()
        chat_main(chat_args)

    if args.command == "dpo":
        # Get the default args for the launch command
        dpo_training_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "dpo.py")
        args = launch_command_parser().parse_args([dpo_training_script])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "dpo"
        launch_command(args)  # launch training

    elif args.command == "env":
        print_env()

    elif args.command == "grpo":
        # Get the default args for the launch command
        grpo_training_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "grpo.py")
        args = launch_command_parser().parse_args([grpo_training_script])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "grpo"
        launch_command(args)  # launch training

    elif args.command == "kto":
        # Get the default args for the launch command
        kto_training_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "kto.py")
        args = launch_command_parser().parse_args([kto_training_script])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "kto"
        launch_command(args)  # launch training

    elif args.command == "sft":
        # Get the default args for the launch command
        sft_training_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "sft.py")
        args = launch_command_parser().parse_args([sft_training_script])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "sft"
        launch_command(args)  # launch training

    elif args.command == "vllm-serve":
        (script_args,) = parser.parse_args_and_config()
        vllm_serve_main(script_args)


if __name__ == "__main__":
    main()
