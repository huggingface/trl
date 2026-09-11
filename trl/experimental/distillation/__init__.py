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

import warnings
from dataclasses import dataclass

from ...trainer import DistillationConfig as _DistillationConfig
from ...trainer import DistillationTrainer as _DistillationTrainer


__all__ = ["DistillationConfig", "DistillationTrainer"]


@dataclass
class DistillationConfig(_DistillationConfig):
    def __post_init__(self):
        warnings.warn(
            "This import path is deprecated and will be removed in v2.0.0. "
            "The `DistillationConfig` has been promoted to the stable API. "
            "Update your imports to `from trl import DistillationConfig`.",
            FutureWarning,
            stacklevel=3,
        )
        super().__post_init__()


@dataclass
class DistillationTrainer(_DistillationTrainer):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This import path is deprecated and will be removed in v2.0.0. "
            "The `DistillationTrainer` has been promoted to the stable API. "
            "Update your imports to `from trl import DistillationTrainer`.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
