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


import time

from rich.console import Console


def chat():
    console = Console()
    # Make sure to import things locally to avoid verbose from third party libs.
    with console.status("[bold purple]Welcome! Initializing the TRL CLI..."):
        time.sleep(1)
        ...
        # from .utils import init_zero_verbose

        # init_zero_verbose()
        # trl_examples_dir = os.path.dirname(__file__)

    # command = f"python {trl_examples_dir}/scripts/chat.py {' '.join(sys.argv[2:])}"

    # try:
    #     subprocess.run(
    #         command.split(),
    #         text=True,
    #         check=True,
    #         encoding="utf-8",
    #         cwd=os.getcwd(),
    #         env=os.environ.copy(),
    #     )
    # except (CalledProcessError, ChildProcessError) as exc:
    #     console.log("TRL - CHAT failed! See the logs above for further details.")
    #     raise ValueError("TRL CLI failed! Check the traceback above..") from exc


if __name__ == "__main__":
    chat()
