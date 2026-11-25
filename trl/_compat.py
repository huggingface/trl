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

"""
Compatibility shims for third-party dependencies.

This module contains temporary patches to handle version incompatibilities between TRL's dependencies.

Each patch should be removed when minimum version requirements eliminate the need.
"""

from .import_utils import _is_package_available


def _patch_vllm_logging() -> None:
    """Set vLLM logging level to ERROR by default to reduce noise."""
    if _is_package_available("vllm"):
        import os

        os.environ["VLLM_LOGGING_LEVEL"] = os.getenv("VLLM_LOGGING_LEVEL", "ERROR")


# Apply vLLM patches
_patch_vllm_logging()
