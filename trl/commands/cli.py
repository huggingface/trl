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
import platform
import subprocess
import sys
from importlib.metadata import version
from subprocess import CalledProcessError

import torch
from accelerate.commands.config import default_config_file, load_config_from_file
from rich.console import Console
from transformers import is_bitsandbytes_available
from transformers.utils import is_liger_kernel_available, is_openai_available, is_peft_available

from .. import __version__, is_deepspeed_available, is_diffusers_available, is_llm_blender_available
from .cli_utils import get_git_commit_hash


SUPPORTED_COMMANDS = ["sft", "dpo", "chat", "kto", "env"]


def print_env():
    if torch.cuda.is_available():
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    accelerate_config = accelerate_config_str = "not found"

    # Get the default from the config file.
    if os.path.isfile(default_config_file):
        accelerate_config = load_config_from_file(default_config_file).to_dict()

    accelerate_config_str = (
        "\n" + "\n".join([f"  - {prop}: {val}" for prop, val in accelerate_config.items()])
        if isinstance(accelerate_config, dict)
        else accelerate_config
    )

    commit_hash = get_git_commit_hash("trl")

    info = {
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": version("torch"),
        "CUDA device(s)": ", ".join(devices) if torch.cuda.is_available() else "not available",
        "Transformers version": version("transformers"),
        "Accelerate version": version("accelerate"),
        "Accelerate config": accelerate_config_str,
        "Datasets version": version("datasets"),
        "HF Hub version": version("huggingface_hub"),
        "TRL version": f"{__version__}+{commit_hash[:7]}" if commit_hash else __version__,
        "bitsandbytes version": version("bitsandbytes") if is_bitsandbytes_available() else "not installed",
        "DeepSpeed version": version("deepspeed") if is_deepspeed_available() else "not installed",
        "Diffusers version": version("diffusers") if is_diffusers_available() else "not installed",
        "Liger-Kernel version": version("liger_kernel") if is_liger_kernel_available() else "not installed",
        "LLM-Blender version": version("llm_blender") if is_llm_blender_available() else "not installed",
        "OpenAI version": version("openai") if is_openai_available() else "not installed",
        "PEFT version": version("peft") if is_peft_available() else "not installed",
    }

    info_str = "\n".join([f"- {prop}: {val}" for prop, val in info.items()])
    print(f"\nCopy-paste the following information when reporting an issue:\n\n{info_str}\n")  # noqa


def train(command_name):
    console = Console()
    # Make sure to import things locally to avoid verbose from third party libs.
    with console.status("[bold purple]Welcome! Initializing the TRL CLI..."):
        from trl.commands.cli_utils import init_zero_verbose

        init_zero_verbose()
        command_name = sys.argv[1]
        trl_examples_dir = os.path.dirname(__file__)

    command = f"accelerate launch {trl_examples_dir}/scripts/{command_name}.py {' '.join(sys.argv[2:])}"

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


def chat():
    console = Console()
    # Make sure to import things locally to avoid verbose from third party libs.
    with console.status("[bold purple]Welcome! Initializing the TRL CLI..."):
        from trl.commands.cli_utils import init_zero_verbose

        init_zero_verbose()
        trl_examples_dir = os.path.dirname(__file__)

    command = f"python {trl_examples_dir}/scripts/chat.py {' '.join(sys.argv[2:])}"

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
        console.log("TRL - CHAT failed! See the logs above for further details.")
        raise ValueError("TRL CLI failed! Check the traceback above..") from exc


def main():
    command_name = sys.argv[1]

    if command_name in ["sft", "dpo", "kto"]:
        train(command_name)
    elif command_name == "chat":
        chat()
    elif command_name == "env":
        print_env()
    else:
        raise ValueError(
            f"Please use one of the supported commands, got {command_name} - supported commands are {SUPPORTED_COMMANDS}"
        )


if __name__ == "__main__":
    main()
