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

from dataclasses import dataclass, field

from trl.trainer.online_dpo_config import OnlineDPOConfig


@dataclass
class NashMDConfig(OnlineDPOConfig):
    r"""
    Configuration class for the [`NashMDTrainer`].

    Subclass of [`OnlineDPOConfig`] we can use all its arguments and add the following:

    Parameters:
        mixture_coef (`float` or `list[float]`, *optional*, defaults to `0.5`):
            Logit mixture coefficient for the model and reference model. If a list of floats is provided then the
            mixture coefficient is selected for each new epoch and the last coefficient is used for the rest of the
            epochs.
    """

    mixture_coef: list[float] = field(
        default_factory=lambda: [0.5],
        metadata={
            "help": "Logit mixture coefficient for the model and reference model. If a list of floats is provided "
            "then the mixture coefficient is selected for each new epoch and the last coefficient is used for the "
            "rest of the epochs."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.mixture_coef, "__len__") and len(self.mixture_coef) == 1:
            self.mixture_coef = self.mixture_coef[0]
