# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import os
import sys

from accelerate.commands.launch import launch_command, launch_command_parser

from trl.scripts.dpo import make_parser as make_dpo_parser


def main():
    parser = argparse.ArgumentParser("TRL CLI", usage="trl", allow_abbrev=False)

    # Add the subparsers
    subparsers = parser.add_subparsers(help="available commands", dest="command")

    # Add the subparsers for every script
    make_dpo_parser(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    if args.command == "dpo":
        # Get the default args for the launch command
        dpo_training_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "dpo.py")
        args = launch_command_parser().parse_args([dpo_training_script])

        # Feed the args to the launch command
        args.training_script_args = sys.argv[2:]  # remove "trl" and "dpo"

        # Launch the training
        launch_command(args)


if __name__ == "__main__":
    main()
