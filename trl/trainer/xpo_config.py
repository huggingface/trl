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
from typing import List, Optional

from trl.trainer.online_dpo_config import OnlineDPOConfig


@dataclass
class XPOConfig(OnlineDPOConfig):
    r"""
    Configuration class for the [`XPOTrainer`].

    Subclass of [`OnlineDPOConfig`] we can use all its arguments and add the following:

    Parameters:
        alpha (`float`, defaults to `1e-5`):
            Weight of the XPO loss term.
        alpha_list (`List[float]`, *optional*, defaults to `None`):
            List of α values to use for each epoch. If a list of floats is provided then the α is selected for each new epoch and the last α is used for the rest of the epochs.
    """

    alpha: float = 1e-5
    alpha_list: Optional[List[float]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.alpha_list is not None:
            self.alpha = self.alpha_list
