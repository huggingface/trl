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
import sys

from transformers import HfArgumentParser

from trl import DPOConfig, ModelConfig, ScriptArguments


def main():
    parser = argparse.ArgumentParser(prog="trl", description="A CLI tool for training and fine-tuning")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=HfArgumentParser)

    # 'dpo' subcommand
    dpo_parser = subparsers.add_parser(
        "dpo", help="Run the DPO training process", dataclass_types=(ScriptArguments, DPOConfig, ModelConfig)
    )

    args = parser.parse_args()
    sys.argv = sys.argv[1:]  # Remove 'trl' from sys.argv

    if args.command == "dpo":
        from trl.scripts.dpo import main as dpo_main

        script_args, training_args, model_config = dpo_parser.parse_args_and_config()
        dpo_main(script_args, training_args, model_config)
