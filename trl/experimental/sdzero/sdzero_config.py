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
class SDZeroConfig(SelfDistillationConfig):
    r"""
    Configuration class for [`SDZeroTrainer`].

    Parameters:
        separator (`str`, *optional*, defaults to `"\n\n"`):
            Text inserted between the student's initial response and the control prompt when composing the
            teacher context. Should match the separator used during Phase 1 data collection.
        teacher_model_kind (`str`, *optional*, defaults to `"base"`):
            Semantic teacher choice. `base` freezes the weights at initialization — pass the Phase 1 SRT
            checkpoint as the `model` argument to use it as the static teacher. `live` and `ema` are
            supported via the base class.
        num_generations (`int`, *optional*, defaults to `1`):
            Number of rollouts sampled per prompt per training step. SD-Zero Phase 2 uses one rollout.
    """

    separator: str = field(
        default="\n\n",
        metadata={"help": "Text inserted between the student response and the control prompt in the teacher context."},
    )
    teacher_model_kind: str = field(
        default="base",
        metadata={"help": "Semantic teacher choice. Defaults to 'base' to freeze the Phase 1 SRT model."},
    )
    num_generations: int = field(
        default=1,
        metadata={"help": "Number of rollouts per prompt. SD-Zero Phase 2 uses 1."},
    )
