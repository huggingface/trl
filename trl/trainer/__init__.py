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

from typing import TYPE_CHECKING

from ..import_utils import _LazyModule


_import_structure = {
    "bco_config": ["BCOConfig"],  # deprecated import
    "bco_trainer": ["BCOTrainer"],  # deprecated import
    "callbacks": [
        "BEMACallback",
        "LogCompletionsCallback",
        "RichProgressCallback",
        "SyncRefModelCallback",
        "WeaveCallback",
        "WinRateCallback",  # deprecated import
    ],
    "cpo_config": ["CPOConfig"],  # deprecated import
    "cpo_trainer": ["CPOTrainer"],  # deprecated import
    "dpo_config": ["DPOConfig", "FDivergenceConstants", "FDivergenceType"],
    "dpo_trainer": ["DPOTrainer"],
    "gkd_config": ["GKDConfig"],  # deprecated import
    "gkd_trainer": ["GKDTrainer"],  # deprecated import
    "grpo_config": ["GRPOConfig"],
    "grpo_trainer": ["GRPOTrainer"],
    "judges": [
        "AllTrueJudge",  # deprecated import
        "BaseBinaryJudge",  # deprecated import
        "BaseJudge",  # deprecated import
        "BasePairwiseJudge",  # deprecated import
        "BaseRankJudge",  # deprecated import
        "HfPairwiseJudge",  # deprecated import
        "OpenAIPairwiseJudge",  # deprecated import
        "PairRMJudge",  # deprecated import
    ],
    "kto_config": ["KTOConfig"],
    "kto_trainer": ["KTOTrainer"],
    "model_config": ["ModelConfig"],
    "nash_md_config": ["NashMDConfig"],  # deprecated import
    "nash_md_trainer": ["NashMDTrainer"],  # deprecated import
    "online_dpo_config": ["OnlineDPOConfig"],  # deprecated import
    "online_dpo_trainer": ["OnlineDPOTrainer"],  # deprecated import
    "orpo_config": ["ORPOConfig"],  # deprecated import
    "orpo_trainer": ["ORPOTrainer"],  # deprecated import
    "ppo_config": ["PPOConfig"],  # deprecated import
    "ppo_trainer": ["PPOTrainer"],  # deprecated import
    "prm_config": ["PRMConfig"],  # deprecated import
    "prm_trainer": ["PRMTrainer"],  # deprecated import
    "reward_config": ["RewardConfig"],
    "reward_trainer": ["RewardTrainer"],
    "rloo_config": ["RLOOConfig"],
    "rloo_trainer": ["RLOOTrainer"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "utils": [
        "RunningMoments",
        "disable_dropout_in_model",
        "empty_cache",
        "ensure_master_addr_port",
        "get_kbit_device_map",
        "get_peft_config",
        "get_quantization_config",
    ],
    "xpo_config": ["XPOConfig"],  # deprecated import
    "xpo_trainer": ["XPOTrainer"],  # deprecated import
}

if TYPE_CHECKING:
    from .bco_config import BCOConfig  # deprecated import
    from .bco_trainer import BCOTrainer  # deprecated import
    from .callbacks import (
        BEMACallback,
        LogCompletionsCallback,
        RichProgressCallback,
        SyncRefModelCallback,
        WeaveCallback,
        WinRateCallback,  # deprecated import
    )
    from .cpo_config import CPOConfig  # deprecated import
    from .cpo_trainer import CPOTrainer  # deprecated import
    from .dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
    from .dpo_trainer import DPOTrainer
    from .gkd_config import GKDConfig  # deprecated import
    from .gkd_trainer import GKDTrainer  # deprecated import
    from .grpo_config import GRPOConfig
    from .grpo_trainer import GRPOTrainer
    from .judges import (
        AllTrueJudge,  # deprecated import
        BaseBinaryJudge,  # deprecated import
        BaseJudge,  # deprecated import
        BasePairwiseJudge,  # deprecated import
        BaseRankJudge,  # deprecated import
        HfPairwiseJudge,  # deprecated import
        OpenAIPairwiseJudge,  # deprecated import
        PairRMJudge,  # deprecated import
    )
    from .kto_config import KTOConfig
    from .kto_trainer import KTOTrainer
    from .model_config import ModelConfig
    from .nash_md_config import NashMDConfig  # deprecated import
    from .nash_md_trainer import NashMDTrainer  # deprecated import
    from .online_dpo_config import OnlineDPOConfig  # deprecated import
    from .online_dpo_trainer import OnlineDPOTrainer  # deprecated import
    from .orpo_config import ORPOConfig  # deprecated import
    from .orpo_trainer import ORPOTrainer  # deprecated import
    from .ppo_config import PPOConfig  # deprecated import
    from .ppo_trainer import PPOTrainer  # deprecated import
    from .prm_config import PRMConfig  # deprecated import
    from .prm_trainer import PRMTrainer  # deprecated import
    from .reward_config import RewardConfig
    from .reward_trainer import RewardTrainer
    from .rloo_config import RLOOConfig
    from .rloo_trainer import RLOOTrainer
    from .sft_config import SFTConfig
    from .sft_trainer import SFTTrainer
    from .utils import (
        RunningMoments,
        disable_dropout_in_model,
        empty_cache,
        ensure_master_addr_port,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )
    from .xpo_config import XPOConfig  # deprecated import
    from .xpo_trainer import XPOTrainer  # deprecated import
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
