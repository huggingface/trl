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

from ..import_utils import suppress_experimental_warning


with suppress_experimental_warning():
    from ..experimental.kto import KTOConfig as _KTOConfig


@dataclass
class KTOConfig(_KTOConfig):
    def __post_init__(self):
        warnings.warn(
            "The `KTOConfig` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.kto import KTOConfig`. For more information, see "
            "https://github.com/huggingface/trl/issues/4223. Promoting KTO to the stable API is a high-priority task. "
            "Until then, this current path (`from trl import KTOConfig`) will remain, but API changes may occur.",
            FutureWarning,
            stacklevel=2,
        )
        super().__post_init__()
