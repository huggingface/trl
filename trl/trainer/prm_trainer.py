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

from ..experimental.prm import PRMTrainer as ExperimentalPRMTrainer


class PRMTrainer(ExperimentalPRMTrainer):
    """
    Initialize PRMTrainer.

    .. warning::
        This class is deprecated and will be removed in TRL 0.29.0. Please use
        `trl.experimental.prm.PRMTrainer` instead. See https://github.com/huggingface/trl/issues/4467
        for more information.

    For full documentation, see [`trl.experimental.prm.PRMTrainer`].
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PRMTrainer is deprecated and will be removed in TRL 0.29.0. "
            "Please use `trl.experimental.prm.PRMTrainer` instead. "
            "See https://github.com/huggingface/trl/issues/4467 for more information.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
