# flake8: noqa

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

# There is a circular import in the PPOTrainer if we let isort sort these
# isort: off
from .utils import (
    AdaptiveKLController,
    FixedKLController,
    ConstantLengthDataset,
    DataCollatorForCompletionOnlyLM,
    RunningMoments,
    disable_dropout_in_model,
    peft_module_casting_to_bf16,
)

# isort: on

from ..import_utils import is_diffusers_available
from .base import BaseTrainer
from .ddpo_config import DDPOConfig


if is_diffusers_available():
    from .ddpo_trainer import DDPOTrainer

from .dpo_trainer import DPOTrainer
from .iterative_sft_trainer import IterativeSFTTrainer
from .model_config import ModelConfig
from .ppo_config import PPOConfig
from .ppo_trainer import PPOTrainer
from .reward_config import RewardConfig
from .reward_trainer import RewardTrainer, compute_accuracy
from .sft_trainer import SFTTrainer
