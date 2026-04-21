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
from typing import Literal

from ..self_distillation.self_distillation_config import SelfDistillationConfig


@dataclass
class SDZeroConfig(SelfDistillationConfig):
    r"""
    Configuration class for [`SDZeroTrainer`].

    This class extends [`experimental.self_distillation.SelfDistillationConfig`] and overrides with parameters and defaults
    relevant for SD-Zero.

    Parameters:
        separator (`str`, *optional*, defaults to `"\n\n"`):
            Text inserted between the student's initial response and the control prompt when composing the
            teacher context. Should match the separator used during SRT training.

         > Parameters that control the teacher

        teacher_model_kind (`str`, *optional*, defaults to `"base"`):
            Semantic teacher choice. Defaults to `"base"` so the teacher stays fixed throughout training.
        teacher_update_rate (`float`, *optional*, defaults to `1.0`):
            EMA update rate used when `teacher_model_kind="ema"`. Defaults to `1.0` so opting into EMA gives
            periodic hard teacher resync.
        teacher_sync_steps (`int`, *optional*, defaults to `512`):
            Number of optimizer steps between EMA teacher updates.

         > Parameters that control the loss

        distillation_mode (`str`, *optional*, defaults to `"full_logits"`):
            Distillation objective mode. Defaults to `"full_logits".
        distillation_alpha (`float`, *optional*, defaults to `1.0`):
            KL direction. Defaults to `1.0`.
        distillation_is_clip (`float` or `None`, *optional*):
            Importance-sampling clip. Defaults to `None`, which disables clipping.

         > Parameters that control the student generation

        num_generations (`int`, *optional*, defaults to `1`):
            Number of rollouts sampled per prompt per training step.
    """

    separator: str = field(
        default="\n\n",
        metadata={
            "help": "Text inserted between the student's initial response and the control prompt when composing "
            "the teacher context. Should match the separator used during Phase 1 data collection."
        },
    )
    teacher_model_kind: str = field(
        default="base",
        metadata={
            "help": "Semantic teacher choice. Defaults to 'base' so the teacher stays fixed throughout training."
        },
    )
    teacher_update_rate: float = field(
        default=1.0,
        metadata={
            "help": "EMA update rate used when teacher_model_kind='ema'. Defaults to 1.0 so opting into EMA "
            "gives periodic hard teacher resync."
        },
    )
    teacher_sync_steps: int = field(
        default=512,
        metadata={"help": "Number of optimizer steps between EMA teacher updates."},
    )
    distillation_mode: Literal["sampled_token", "full_logits", "topk_logits"] = field(
        default="full_logits",
        metadata={"help": "Distillation objective mode. Defaults to 'full_logits'."},
    )
    distillation_alpha: float = field(
        default=1.0,
        metadata={"help": "KL direction. Defaults to 1.0."},
    )
    distillation_is_clip: float | None = field(
        default=None,
        metadata={"help": "Importance-sampling clip. Defaults to `None`, which disables clipping."},
    )
    num_generations: int = field(
        default=1,
        metadata={"help": "Number of rollouts sampled per prompt per training step."},
    )
