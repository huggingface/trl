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
from functools import wraps

import pytest
import torch


@pytest.fixture(autouse=True)
def set_model_float32_dtype(monkeypatch):
    """Auto-inject float32 dtype for tiny models defined in trl-internal-testing."""
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

    def create_classmethod_wrapper(original_classmethod):
        # Extract the underlying function from the classmethod
        original_func = original_classmethod.__func__

        @wraps(original_func)
        def wrapper(cls, pretrained_model_name_or_path, *args, **kwargs):
            # Only inject if model_id is one of trl-internal-testing
            if (
                isinstance(pretrained_model_name_or_path, str)
                and "trl-internal-testing" in pretrained_model_name_or_path
            ):
                if "dtype" not in kwargs:
                    kwargs["dtype"] = "float32"

            return original_func(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Re-wrap as classmethod
        return classmethod(wrapper)

    # Patch base classes - this affects all models, tokenizers, and processors
    for cls in [
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    ]:
        monkeypatch.setattr(cls, "from_pretrained", create_classmethod_wrapper(cls.from_pretrained))


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
