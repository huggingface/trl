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

from dataclasses import dataclass, field

from ...trainer.grpo_config import GRPOConfig as _GRPOConfig


@dataclass
class GFPOConfig(_GRPOConfig):
    num_remains_in_group: int | None = field(
        default=None,
        metadata={
            "help": "number inputs remains after group filter function, `'num_remains_in_group'` must be >=2 if given."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.num_remains_in_group is not None and self.num_remains_in_group >= self.num_generations:
            raise ValueError(
                f"Number remains in Group {self.num_remains_in_group} must be less than num_generations : {self.num_generations}."
            )
