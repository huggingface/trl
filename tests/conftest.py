# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """
    Automatically cleanup GPU memory after each test.

    This fixture helps prevent CUDA out of memory errors when running tests in parallel with pytest-xdist by ensuring
    models and tensors are properly garbage collected and GPU memory caches are cleared between tests.
    """
    yield
    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
