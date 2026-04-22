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

from ...trainer.sft_config import SFTConfig


@dataclass
class SRTConfig(SFTConfig):
    r"""
    Configuration class for [`SRTTrainer`].

    Parameters:
        assistant_turn_template (`str`, *optional*, defaults to `"{y_init}\n\n{control_prompt}\n\n{y_revised}"`):
            Template used to compose the assistant turn from the initial answer, control prompt, and revised answer.
            Must end with `{y_revised}` so the revised-answer supervision boundary is well-defined.
        include_generation_loss (`bool`, *optional*, defaults to `True`):
            Whether to include the generation loss term, which supervises the model on the full assistant turn
            (initial answer, control prompt, and revised answer) given only the problem.
        include_revision_loss (`bool`, *optional*, defaults to `True`):
            Whether to include the revision loss term, which supervises the model only on the revised answer given
            the full context (problem, initial answer, control prompt).
    """

    assistant_turn_template: str = field(
        default="{y_init}\n\n{control_prompt}\n\n{y_revised}",
        metadata={
            "help": "Template used to compose the assistant turn from the initial answer, control prompt, and revised "
            "answer. Must end with `{y_revised}` so the revised-answer supervision boundary is well-defined."
        },
    )
    include_generation_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to include the generation loss term, which supervises the model on the full assistant turn "
            "(initial answer, control prompt, and revised answer) given only the problem"
        },
    )
    include_revision_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to include the revision loss term, which supervises the model only on the revised "
            "answer given the full context (problem, initial answer, control prompt)."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.assistant_turn_template.endswith("{y_revised}"):
            raise ValueError("`assistant_turn_template` must end with `{y_revised}`.")
        if not (self.include_revision_loss or self.include_generation_loss):
            raise ValueError("At least one of `include_revision_loss` or `include_generation_loss` must be True.")
