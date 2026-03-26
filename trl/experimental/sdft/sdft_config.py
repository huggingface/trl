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

from ..self_distillation.self_distillation_config import SelfDistillationConfig


@dataclass
class SDFTConfig(SelfDistillationConfig):
    r"""
    Configuration class for [`SDFTTrainer`].

    This adapts the official SDFT implementation to the TRL trainer API while reusing the common self-distillation
    configuration shared with SDPO.

    Parameters:
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the student and teacher models.
        generate_from_teacher (`bool`, *optional*, defaults to `False`):
            Whether on-policy generation should use the teacher-conditioned prompt instead of the student prompt.
        teacher_prompt_template (`str`, *optional*, defaults to `"{prompt}\n\n{privileged_context}"`):
            Template used to combine the student prompt and privileged context into the teacher prompt.
        num_loss_tokens_to_skip (`int`, *optional*, defaults to `0`):
            Number of initial completion tokens to exclude from the distillation loss.
    """

    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the student and teacher models."},
    )
    generate_from_teacher: bool = field(
        default=False,
        metadata={"help": "Whether on-policy generation should use the teacher-conditioned prompt."},
    )
    teacher_prompt_template: str = field(
        default="{prompt}\n\n{privileged_context}",
        metadata={
            "help": "Template used to combine the student prompt and privileged context into the teacher prompt."
        },
    )
    num_loss_tokens_to_skip: int = field(
        default=0,
        metadata={"help": "Number of initial completion tokens to exclude from the distillation loss."},
    )

    def __post_init__(self):
        super().__post_init__()
        if (
            "{prompt}" not in self.teacher_prompt_template
            or "{privileged_context}" not in self.teacher_prompt_template
        ):
            raise ValueError(
                "teacher_prompt_template must contain both `{prompt}` and `{privileged_context}` placeholders"
            )
        if self.num_loss_tokens_to_skip < 0:
            raise ValueError("num_loss_tokens_to_skip must be non-negative")
