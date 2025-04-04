# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule

from ..import_utils import is_diffusers_available


_import_structure = {
    "alignprop_config": ["AlignPropConfig"],
    "alignprop_trainer": ["AlignPropTrainer"],
    "bco_config": ["BCOConfig"],
    "bco_trainer": ["BCOTrainer"],
    "cpo_config": ["CPOConfig"],
    "cpo_trainer": ["CPOTrainer"],
    "dpo_config": ["DPOConfig", "FDivergenceConstants", "FDivergenceType"],
    "dpo_trainer": ["DPOTrainer"],
    "gkd_config": ["GKDConfig"],
    "gkd_trainer": ["GKDTrainer"],
    "grpo_config": ["GRPOConfig"],
    "grpo_trainer": ["GRPOTrainer"],
    "iterative_sft_trainer": ["IterativeSFTTrainer"],
    "kto_config": ["KTOConfig"],
    "kto_trainer": ["KTOTrainer"],
    "model_config": ["ModelConfig"],
    "nash_md_config": ["NashMDConfig"],
    "nash_md_trainer": ["NashMDTrainer"],
    "online_dpo_config": ["OnlineDPOConfig"],
    "online_dpo_trainer": ["OnlineDPOTrainer"],
    "orpo_config": ["ORPOConfig"],
    "orpo_trainer": ["ORPOTrainer"],
    "ppo_config": ["PPOConfig"],
    "ppo_trainer": ["PPOTrainer"],
    "prm_config": ["PRMConfig"],
    "prm_trainer": ["PRMTrainer"],
    "reward_config": ["RewardConfig"],
    "reward_trainer": ["RewardTrainer"],
    "rloo_config": ["RLOOConfig"],
    "rloo_trainer": ["RLOOTrainer"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "xpo_config": ["XPOConfig"],
    "xpo_trainer": ["XPOTrainer"],
}
try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["ddpo_trainer"] = ["DDPOTrainer"]
    _import_structure["ddpo_config"] = ["DDPOConfig"]

if TYPE_CHECKING:
    from .alignprop_config import AlignPropConfig
    from .alignprop_trainer import AlignPropTrainer
    from .bco_config import BCOConfig
    from .bco_trainer import BCOTrainer
    from .cpo_config import CPOConfig
    from .cpo_trainer import CPOTrainer
    from .dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
    from .dpo_trainer import DPOTrainer
    from .gkd_config import GKDConfig
    from .gkd_trainer import GKDTrainer
    from .grpo_config import GRPOConfig
    from .grpo_trainer import GRPOTrainer
    from .iterative_sft_trainer import IterativeSFTTrainer
    from .kto_config import KTOConfig
    from .kto_trainer import KTOTrainer
    from .model_config import ModelConfig
    from .nash_md_config import NashMDConfig
    from .nash_md_trainer import NashMDTrainer
    from .online_dpo_config import OnlineDPOConfig
    from .online_dpo_trainer import OnlineDPOTrainer
    from .orpo_config import ORPOConfig
    from .orpo_trainer import ORPOTrainer
    from .ppo_config import PPOConfig
    from .ppo_trainer import PPOTrainer
    from .prm_config import PRMConfig
    from .prm_trainer import PRMTrainer
    from .reward_config import RewardConfig
    from .reward_trainer import RewardTrainer
    from .rloo_config import RLOOConfig
    from .rloo_trainer import RLOOTrainer
    from .sft_config import SFTConfig
    from .sft_trainer import SFTTrainer
    from .xpo_config import XPOConfig
    from .xpo_trainer import XPOTrainer

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .ddpo_config import DDPOConfig
        from .ddpo_trainer import DDPOTrainer
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
