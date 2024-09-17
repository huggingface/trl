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

from transformers import is_wandb_available

from trl import is_diffusers_available, is_liger_kernel_available


def require_diffusers(test_case):
    """
    Decorator marking a test that requires diffusers. Skips the test if diffusers is not available.
    """
    return unittest.skipUnless(is_diffusers_available(), "test requires diffusers")(test_case)


def require_no_wandb(test_case):
    """
    Decorator marking a test that requires no wandb. Skips the test if wandb is available.
    """
    return unittest.skipUnless(not is_wandb_available(), "test requires no wandb")(test_case)


def require_liger_kernel(test_case):
    """
    Decorator marking a test that requires liger_kernel. Skips the test if liger_kernel is not available.
    """
    return unittest.skipUnless(is_liger_kernel_available(), "test requires liger_kernel")(test_case)
