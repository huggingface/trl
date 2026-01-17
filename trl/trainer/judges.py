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

from ..import_utils import suppress_experimental_warning


with suppress_experimental_warning():
    from ..experimental.judges import AllTrueJudge as _AllTrueJudge
    from ..experimental.judges import BaseBinaryJudge as _BaseBinaryJudge
    from ..experimental.judges import BaseJudge as _BaseJudge
    from ..experimental.judges import BasePairwiseJudge as _BasePairwiseJudge
    from ..experimental.judges import BaseRankJudge as _BaseRankJudge
    from ..experimental.judges import HfPairwiseJudge as _HfPairwiseJudge
    from ..experimental.judges import OpenAIPairwiseJudge as _OpenAIPairwiseJudge
    from ..experimental.judges import PairRMJudge as _PairRMJudge


class AllTrueJudge(_AllTrueJudge):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `AllTrueJudge` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.judges import AllTrueJudge`. The current import path will be removed and no "
            "longer supported in TRL 0.29.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class BaseBinaryJudge(_BaseBinaryJudge):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `BaseBinaryJudge` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.judges import BaseBinaryJudge`. The current import path will be removed and no "
            "longer supported in TRL 0.29.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class BaseJudge(_BaseJudge):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `BaseJudge` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.judges import BaseJudge`. The current import path will be removed and no "
            "longer supported in TRL 0.29.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class BasePairwiseJudge(_BasePairwiseJudge):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `BasePairwiseJudge` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.judges import BasePairwiseJudge`. The current import path will be removed and no "
            "longer supported in TRL 0.29.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class BaseRankJudge(_BaseRankJudge):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `BaseRankJudge` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.judges import BaseRankJudge`. The current import path will be removed and no "
            "longer supported in TRL 0.29.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class HfPairwiseJudge(_HfPairwiseJudge):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `HfPairwiseJudge` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.judges import HfPairwiseJudge`. The current import path will be removed and no "
            "longer supported in TRL 0.29.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class OpenAIPairwiseJudge(_OpenAIPairwiseJudge):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `OpenAIPairwiseJudge` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.judges import OpenAIPairwiseJudge`. The current import path will be removed and no "
            "longer supported in TRL 0.29.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class PairRMJudge(_PairRMJudge):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `PairRMJudge` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.judges import PairRMJudge`. The current import path will be removed and no "
            "longer supported in TRL 0.29.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
