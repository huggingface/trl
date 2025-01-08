# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field

from trl.trainer.online_dpo_config import OnlineDPOConfig


@dataclass
class XPOConfig(OnlineDPOConfig):
    r"""
    Configuration class for the [`XPOTrainer`].

    Subclass of [`OnlineDPOConfig`] we can use all its arguments and add the following:

    Parameters:
        alpha (`float` or `list[float]`, *optional*, defaults to `1e-5`):
            Weight of the XPO loss term. If a list of floats is provided then the alpha is selected for each new epoch
            and the last alpha is used for the rest of the epochs.
    """

    alpha: list[float] = field(
        default_factory=lambda: [1e-5],
        metadata={
            "help": "Weight of the XPO loss term. If a list of floats is provided then the alpha is selected for each "
            "new epoch and the last alpha is used for the rest of the epochs."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.alpha, "__len__") and len(self.alpha) == 1:
            self.alpha = self.alpha[0]
