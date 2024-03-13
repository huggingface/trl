# This file is a copy of trl/examples/scripts/sft.py so that we could
# use it together with rich and the TRL CLI in a more customizable manner.
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
import os
import subprocess
from subprocess import CalledProcessError

from rich.console import Console

from trl.commands.config_parser import YamlConfigParser


SUPPORTED_COMMANDS = ["sft"]


def main():
    console = Console()
    # Make sure to import things locally to avoid verbose from third party libs.
    with console.status("[bold purple]Welcome! Initializing the TRL CLI..."):
        from trl.commands.cli_utils import SftScriptArguments, init_zero_verbose

        init_zero_verbose()

        from transformers import HfArgumentParser, TrainingArguments

        from trl import ModelConfig

        parser = HfArgumentParser((SftScriptArguments, TrainingArguments, ModelConfig))

        (args, training_args, model_config, command_name) = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )

        command_name = command_name[0]

        if command_name not in SUPPORTED_COMMANDS:
            raise ValueError(
                f"Please use one of the supported commands, got {command_name} - supported commands are {SUPPORTED_COMMANDS}"
            )

        # Get the required args
        config = args.config

        # if the configuration is None, create a new `output_dir` variable
        config_parser = YamlConfigParser(config, [args, training_args, model_config])
        current_dir = os.path.dirname(__file__)

        model_name = model_config.model_name_or_path

    command = f"""
    python {current_dir}/{command_name}.py {config_parser.to_string()}
    """

    try:
        subprocess.run(
            command.split(),
            text=True,
            check=True,
            encoding="utf-8",
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
    except (CalledProcessError, ChildProcessError):
        console.log(f"TRL - {command_name.upper()} failed on {model_name}! See the logs above for further details.")


if __name__ == "__main__":
    main()
