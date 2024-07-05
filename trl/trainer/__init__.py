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
from typing import TYPE_CHECKING
from ..import_utils import _LazyModule, is_diffusers_available, OptionalDependencyNotAvailable

_import_structure = {
    "utils": [
        "AdaptiveKLController",
        "FixedKLController",
        "ConstantLengthDataset",
        "DataCollatorForCompletionOnlyLM",
        "RunningMoments",
        "disable_dropout_in_model",
        "peft_module_casting_to_bf16",
        "RichProgressCallback",
    ],
    "dpo_config": ["DPOConfig", "FDivergenceConstants", "FDivergenceType"],
    "dpo_trainer": ["DPOTrainer"],
    "cpo_config": ["CPOConfig"],
    "cpo_trainer": ["CPOTrainer"],
    "alignprop_config": ["AlignPropConfig"],
    "alignprop_trainer": ["AlignPropTrainer"],
    "iterative_sft_trainer": ["IterativeSFTTrainer"],
    "kto_config": ["KTOConfig"],
    "kto_trainer": ["KTOTrainer"],
    "model_config": ["ModelConfig"],
    "orpo_config": ["ORPOConfig"],
    "orpo_trainer": ["ORPOTrainer"],
    "ppo_config": ["PPOConfig"],
    "ppo_trainer": ["PPOTrainer"],
    "reward_config": ["RewardConfig"],
    "reward_trainer": ["RewardTrainer", "compute_accuracy"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "base": ["BaseTrainer"],
    "ddpo_config": ["DDPOConfig"],
}

try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["ddpo_trainer"] = ["DDPOTrainer"]


if TYPE_CHECKING:
    # isort: off
    from .utils import (
        AdaptiveKLController,
        FixedKLController,
        ConstantLengthDataset,
        DataCollatorForCompletionOnlyLM,
        RunningMoments,
        disable_dropout_in_model,
        peft_module_casting_to_bf16,
        RichProgressCallback,
    )

    # isort: on

    from .base import BaseTrainer
    from .ddpo_config import DDPOConfig

    from .dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
    from .dpo_trainer import DPOTrainer
    from .iterative_sft_trainer import IterativeSFTTrainer
    from .cpo_config import CPOConfig
    from .cpo_trainer import CPOTrainer
    from .alignprop_config import AlignPropConfig
    from .kto_config import KTOConfig
    from .kto_trainer import KTOTrainer
    from .model_config import ModelConfig
    from .orpo_config import ORPOConfig
    from .orpo_trainer import ORPOTrainer
    from .ppo_config import PPOConfig
    from .ppo_trainer import PPOTrainer
    from .reward_config import RewardConfig
    from .reward_trainer import RewardTrainer, compute_accuracy
    from .sft_config import SFTConfig
    from .sft_trainer import SFTTrainer

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .ddpo_trainer import DDPOTrainer
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
