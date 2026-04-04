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
class SSDConfig(SelfDistillationConfig):
    r"""
    Configuration class for [`SSDTrainer`].

    Implements Simple Self-Distillation (SSD) from [*Embarrassingly Simple Self-Distillation Improves Code
    Generation*](https://huggingface.co/papers/2604.01193). SSD samples completions from the model at a training-time
    temperature and truncation configuration, then fine-tunes on those raw, unverified samples with standard
    cross-entropy loss.

    The inherited `temperature`, `top_k`, and `top_p` control the training-time sampling configuration (T_train,
    rho_train in the paper). The evaluation-time configuration (T_eval, rho_eval) is set independently at inference
    time.

    Parameters:
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model during training.
        filter_empty (`bool`, *optional*, defaults to `True`):
            Whether to filter out empty or single-line stub completions from the generated data.
    """

    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model during training."},
    )
    filter_empty: bool = field(
        default=True,
        metadata={"help": "Whether to filter out empty or single-line stub completions."},
    )

    def __post_init__(self):
        super().__post_init__()
