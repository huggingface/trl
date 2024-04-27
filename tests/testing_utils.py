# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest

import torch

from trl import (
    is_bitsandbytes_available,
    is_diffusers_available,
    is_peft_available,
    is_pil_available,
    is_wandb_available,
    is_xpu_available,
)


def require_peft(test_case):
    """
    Decorator marking a test that requires peft. Skips the test if peft is not available.
    """
    if not is_peft_available():
        test_case = unittest.skip("test requires peft")(test_case)
    return test_case


def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires bnb. Skips the test if bnb is not available.
    """
    if not is_bitsandbytes_available():
        test_case = unittest.skip("test requires bnb")(test_case)
    return test_case


def require_diffusers(test_case):
    """
    Decorator marking a test that requires diffusers. Skips the test if diffusers is not available.
    """
    if not is_diffusers_available():
        test_case = unittest.skip("test requires diffusers")(test_case)
    return test_case


def requires_pil(test_case):
    """
    Decorator marking a test that requires PIL. Skips the test if pil is not available.
    """
    if not is_pil_available():
        test_case = unittest.skip("test requires PIL")(test_case)
    return test_case


def require_wandb(test_case, required: bool = True):
    """
    Decorator marking a test that requires wandb. Skips the test if wandb is not available.
    """
    # XOR, i.e.:
    # skip if available and required = False and
    # skip if not available and required = True
    if is_wandb_available() ^ required:
        test_case = unittest.skip("test requires wandb")(test_case)
    return test_case


def require_no_wandb(test_case):
    """
    Decorator marking a test that requires no wandb. Skips the test if wandb is available.
    """
    return require_wandb(test_case, required=False)


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires multiple GPUs. Skips the test if there aren't enough GPUs.
    """
    if torch.cuda.device_count() < 2:
        test_case = unittest.skip("test requires multiple GPUs")(test_case)
    return test_case


def require_torch_gpu(test_case):
    """
    Decorator marking a test that requires GPUs. Skips the test if there is no GPU.
    """
    if not torch.cuda.is_available():
        test_case = unittest.skip("test requires GPU")(test_case)
    return test_case


def require_torch_multi_xpu(test_case):
    """
    Decorator marking a test that requires multiple XPUs. Skips the test if there aren't enough XPUs.
    """
    if torch.xpu.device_count() < 2 and is_xpu_available():
        test_case = unittest.skip("test requires multiple XPUs")(test_case)
    return test_case
