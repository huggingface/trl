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
import platform
from importlib.metadata import version

import torch
from accelerate.commands.config import default_config_file, load_config_from_file
from transformers import is_bitsandbytes_available
from transformers.utils import is_liger_kernel_available, is_openai_available, is_peft_available

from .. import __version__
from ..import_utils import is_deepspeed_available, is_diffusers_available, is_llm_blender_available, is_vllm_available
from .utils import get_git_commit_hash


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
        "TRL version": f"{__version__}+{commit_hash[:7]}" if commit_hash else __version__,
        "PyTorch version": version("torch"),
        "CUDA device(s)": ", ".join(devices) if torch.cuda.is_available() else "not available",
        "Transformers version": version("transformers"),
        "Accelerate version": version("accelerate"),
        "Accelerate config": accelerate_config_str,
        "Datasets version": version("datasets"),
        "HF Hub version": version("huggingface_hub"),
        "bitsandbytes version": version("bitsandbytes") if is_bitsandbytes_available() else "not installed",
        "DeepSpeed version": version("deepspeed") if is_deepspeed_available() else "not installed",
        "Diffusers version": version("diffusers") if is_diffusers_available() else "not installed",
        "Liger-Kernel version": version("liger_kernel") if is_liger_kernel_available() else "not installed",
        "LLM-Blender version": version("llm_blender") if is_llm_blender_available() else "not installed",
        "OpenAI version": version("openai") if is_openai_available() else "not installed",
        "PEFT version": version("peft") if is_peft_available() else "not installed",
        "vLLM version": version("vllm") if is_vllm_available() else "not installed",
    }

    info_str = "\n".join([f"- {prop}: {val}" for prop, val in info.items()])
    print(f"\nCopy-paste the following information when reporting an issue:\n\n{info_str}\n")  # noqa


if __name__ == "__main__":
    print_env()
