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
    "callbacks": ["RichProgressCallback", "SyncRefModelCallback"],
    "utils": [
        "AdaptiveKLController",
        "FixedKLController",
        "ConstantLengthDataset",
        "DataCollatorForCompletionOnlyLM",
        "RunningMoments",
        "disable_dropout_in_model",
        "peft_module_casting_to_bf16",
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
    "bco_config": ["BCOConfig"],
    "bco_trainer": ["BCOTrainer"],
    "model_config": ["ModelConfig"],
    "online_dpo_config": ["OnlineDPOConfig"],
    "online_dpo_trainer": ["OnlineDPOTrainer"],
    "xpo_config": ["XPOConfig"],
    "xpo_trainer": ["XPOTrainer"],
    "orpo_config": ["ORPOConfig"],
    "orpo_trainer": ["ORPOTrainer"],
    "ppo_config": ["PPOConfig"],
    "ppo_trainer": ["PPOTrainer"],
    "ppov2_config": ["PPOv2Config"],
    "ppov2_trainer": ["PPOv2Trainer"],
    "reward_config": ["RewardConfig"],
    "reward_trainer": ["RewardTrainer", "compute_accuracy"],
    "rloo_config": ["RLOOConfig"],
    "rloo_trainer": ["RLOOTrainer"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "base": ["BaseTrainer"],
    "ddpo_config": ["DDPOConfig"],
    "gkd_trainer": ["GKDTrainer"],
    "gkd_config": ["GKDConfig"],
    "callbacks": ["RichProgressCallback", "SyncRefModelCallback", "WinRateCallback", "LogCompletionsCallback"],
    "judges": [
        "BaseJudge",
        "BaseRankJudge",
        "BasePairwiseJudge",
        "RandomRankJudge",
        "RandomPairwiseJudge",
        "PairRMJudge",
        "HfPairwiseJudge",
        "OpenAIPairwiseJudge",
    ],
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
    from .callbacks import RichProgressCallback, SyncRefModelCallback
    from .utils import (
        AdaptiveKLController,
        FixedKLController,
        ConstantLengthDataset,
        DataCollatorForCompletionOnlyLM,
        RunningMoments,
        disable_dropout_in_model,
        peft_module_casting_to_bf16,
        empty_cache,
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
    from .alignprop_trainer import AlignPropTrainer
    from .kto_config import KTOConfig
    from .kto_trainer import KTOTrainer
    from .bco_config import BCOConfig
    from .bco_trainer import BCOTrainer
    from .model_config import ModelConfig
    from .online_dpo_config import OnlineDPOConfig
    from .online_dpo_trainer import OnlineDPOTrainer
    from .xpo_config import XPOConfig
    from .xpo_trainer import XPOTrainer
    from .orpo_config import ORPOConfig
    from .orpo_trainer import ORPOTrainer
    from .ppo_config import PPOConfig
    from .ppo_trainer import PPOTrainer
    from .ppov2_config import PPOv2Config
    from .ppov2_trainer import PPOv2Trainer
    from .reward_config import RewardConfig
    from .reward_trainer import RewardTrainer, compute_accuracy
    from .rloo_config import RLOOConfig
    from .rloo_trainer import RLOOTrainer
    from .sft_config import SFTConfig
    from .sft_trainer import SFTTrainer
    from .gkd_trainer import GKDTrainer
    from .gkd_config import GKDConfig
    from .callbacks import RichProgressCallback, SyncRefModelCallback, WinRateCallback, LogCompletionsCallback
    from .judges import (
        BaseJudge,
        BaseRankJudge,
        BasePairwiseJudge,
        RandomRankJudge,
        RandomPairwiseJudge,
        PairRMJudge,
        HfPairwiseJudge,
        OpenAIPairwiseJudge,
    )

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
