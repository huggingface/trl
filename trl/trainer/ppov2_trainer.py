# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from .ppo_trainer import PPOTrainer


# Define an alias for PPOv2Trainer that raises a warning
class PPOv2Trainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`PPOv2Trainer` is deprecated and has been renamed to `PPOTrainer`. Please use `PPOTrainer` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
