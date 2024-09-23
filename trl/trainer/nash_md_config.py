# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import List, Union

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

    mixture_coef: Union[float, List[float]] = 0.5
