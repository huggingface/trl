# coding=utf-8
# coding=utf-8
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
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class RewardTrainingArguments(TrainingArguments):
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."
        },
    )
