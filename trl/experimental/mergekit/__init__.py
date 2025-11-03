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

"""
Mergekit integration for TRL (Experimental).

This module contains utilities for merging models using mergekit.
This is an experimental feature and may be removed in future versions.
"""

from .callbacks import MergeModelCallback
from .mergekit_utils import MergeConfig, merge_models, upload_model_to_hf


__all__ = [
    "MergeConfig",
    "MergeModelCallback",
    "merge_models",
    "upload_model_to_hf",
]
