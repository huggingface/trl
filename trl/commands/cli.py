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
import sys
from subprocess import CalledProcessError

from rich.console import Console


SUPPORTED_COMMANDS = ["sft", "dpo", "chat"]


def main():
    console = Console()
    # Make sure to import things locally to avoid verbose from third party libs.
    with console.status("[bold purple]Welcome! Initializing the TRL CLI..."):
        from trl.commands.cli_utils import init_zero_verbose

        init_zero_verbose()

        command_name = sys.argv[1]

        if command_name not in SUPPORTED_COMMANDS:
            raise ValueError(
                f"Please use one of the supported commands, got {command_name} - supported commands are {SUPPORTED_COMMANDS}"
            )

        trl_examples_dir = os.path.dirname(__file__)

    # Force-use rich if the `TRL_USE_RICH` env var is not set
    if "TRL_USE_RICH" not in os.environ:
        os.environ["TRL_USE_RICH"] = "1"

    if command_name == "chat":
        command = f"""
        python {trl_examples_dir}/scripts/{command_name}.py {" ".join(sys.argv[2:])}
        """
    else:
        command = f"""
        accelerate launch {trl_examples_dir}/scripts/{command_name}.py {" ".join(sys.argv[2:])}
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
    except (CalledProcessError, ChildProcessError) as exc:
        console.log(f"TRL - {command_name.upper()} failed on ! See the logs above for further details.")
        raise ValueError("TRL CLI failed! Check the traceback above..") from exc


if __name__ == "__main__":
    main()
