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

from trl import is_peft_available


def require_peft(test_case):
    """
    Decorator marking a test that requires peft. Skips the test if peft is not available.
    """
    if not is_peft_available():
        test_case = unittest.skip("test requires peft")(test_case)
    return test_case


def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires bitsandbytes. Skips the test if bitsandbytes is not available.
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        test_case = unittest.skip("test requires bitsandbytes")(test_case)
    return test_case


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires multiple GPUs. Skips the test if there aren't enough GPUs.
    """
    if torch.cuda.device_count() < 2:
        test_case = unittest.skip("test requires multiple GPUs")(test_case)
    return test_case
