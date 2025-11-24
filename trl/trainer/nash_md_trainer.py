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

import warnings
from dataclasses import dataclass

from ..experimental.nash_md import NashMDTrainer as _NashMDTrainer


@dataclass
class NashMDTrainer(_NashMDTrainer):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `NashMDTrainer` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.nash_md import NashMDTrainer`. The current import path will be removed and no "
            "longer supported in TRL 0.29. For more information, see https://github.com/huggingface/trl/issues/4223."
        )
        super().__init__(*args, **kwargs)
