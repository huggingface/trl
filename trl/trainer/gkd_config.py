# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Dict, Optional

from .sft_config import SFTConfig


@dataclass
class GKDConfig(SFTConfig):
    """
    Configuration class for GKDTrainer.

    Args:
        temperature (:obj:`float`, `optional`, defaults to 1.0):
            The temperature to use in the generation when sampling from the model.
        lmbda (:obj:`float`, `optional`, defaults to 0.5):
            The parameter that  that controls the student data fraction (i.e., the proportion of on-policy student-generated outputs).
        beta (:obj:`float`, `optional`, defaults to 0.5):
            Interpolation coefficient between 0 and 1 of the Generalized Jensen-Shannon Divergence loss. When beta is 0, the loss is the KL divergence. When beta is 1, the loss is the Inverse KL Divergence.
        max_new_tokens_response (:obj:`int`, `optional`, defaults to 128):
            The maximum number of tokens to generate in the response.
        teacher_model_name_or_path (:obj:`str`, `optional`, defaults to `None`):
            The model name or path of the teacher model. If `None`, the teacher model will be the same as the model
            being trained.
        teacher_model_init_kwargs (:obj:`Dict`, `optional`, defaults to `None`):
            The initialization kwargs to use when creating the teacher model. If `None`, the teacher model will be
            instantiated with the default initialization kwargs.
        disable_dropout (:obj:`bool`, `optional`, defaults to `True`):
            Whether to disable dropout in the model being trained.
    """
    temperature: float = 1.0
    lmbda: float = 0.5
    beta: float = 0.5
    max_new_tokens_response: int = 128
    teacher_model_name_or_path: Optional[str] = None
    teacher_model_init_kwargs: Optional[Dict] = None
    disable_dropout: bool = True
