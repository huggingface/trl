# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import importlib
import sys


if sys.version_info < (3, 8):
    _is_python_greater_3_8 = False
else:
    _is_python_greater_3_8 = True


def is_peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None


def is_unsloth_available() -> bool:
    return importlib.util.find_spec("unsloth") is not None


def is_accelerate_greater_20_0() -> bool:
    if _is_python_greater_3_8:
        from importlib.metadata import version

        accelerate_version = version("accelerate")
    else:
        import pkg_resources

        accelerate_version = pkg_resources.get_distribution("accelerate").version
    return accelerate_version >= "0.20.0"


def is_transformers_greater_than(version: str) -> bool:
    _transformers_version = importlib.metadata.version("transformers")
    return _transformers_version > version


def is_torch_greater_2_0() -> bool:
    if _is_python_greater_3_8:
        from importlib.metadata import version

        torch_version = version("torch")
    else:
        import pkg_resources

        torch_version = pkg_resources.get_distribution("torch").version
    return torch_version >= "2.0"


def is_diffusers_available() -> bool:
    return importlib.util.find_spec("diffusers") is not None


def is_bitsandbytes_available() -> bool:
    import torch

    # bnb can be imported without GPU but is not usable.
    return importlib.util.find_spec("bitsandbytes") is not None and torch.cuda.is_available()


def is_torchvision_available() -> bool:
    return importlib.util.find_spec("torchvision") is not None


def is_rich_available() -> bool:
    return importlib.util.find_spec("rich") is not None


def is_wandb_available() -> bool:
    return importlib.util.find_spec("wandb") is not None


def is_xpu_available() -> bool:
    if is_accelerate_greater_20_0():
        import accelerate

        return accelerate.utils.is_xpu_available()
    else:
        if importlib.util.find_spec("intel_extension_for_pytorch") is None:
            return False
        try:
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except RuntimeError:
            return False


def is_npu_available() -> bool:
    """Checks if `torch_npu` is installed and potentially if a NPU is in the environment"""
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    return hasattr(torch, "npu") and torch.npu.is_available()
