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

# /// script
# dependencies = [
#     "trl",
# ]
# ///

import platform
from importlib.metadata import version


def print_env():
    import torch
    from transformers import is_bitsandbytes_available
    from transformers.utils import is_peft_available

    from trl import __version__
    from trl.import_utils import is_deepspeed_available, is_liger_kernel_available, is_vllm_available
    from trl.scripts.utils import get_git_commit_hash

    devices = None
    if torch.cuda.is_available():
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        devices = ["MPS"]
    elif torch.xpu.is_available():
        devices = [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]

    commit_hash = get_git_commit_hash("trl")

    info = {
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "TRL version": f"{__version__}+{commit_hash[:7]}" if commit_hash else __version__,
        "PyTorch version": version("torch"),
        "accelerator(s)": ", ".join(devices) if devices is not None else "cpu",
        "Transformers version": version("transformers"),
        "Accelerate version": version("accelerate"),
        "Datasets version": version("datasets"),
        "HF Hub version": version("huggingface_hub"),
        "bitsandbytes version": version("bitsandbytes") if is_bitsandbytes_available() else "not installed",
        "DeepSpeed version": version("deepspeed") if is_deepspeed_available() else "not installed",
        "Liger-Kernel version": version("liger_kernel") if is_liger_kernel_available() else "not installed",
        "PEFT version": version("peft") if is_peft_available() else "not installed",
        "vLLM version": version("vllm") if is_vllm_available() else "not installed",
    }

    info_str = "\n".join([f"- {prop}: {val}" for prop, val in info.items()])
    print(f"\nCopy-paste the following information when reporting an issue:\n\n{info_str}\n")  # noqa


if __name__ == "__main__":
    print_env()
