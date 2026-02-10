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
    "callbacks": [
        "BEMACallback",
        "LogCompletionsCallback",
        "RichProgressCallback",
        "SyncRefModelCallback",
        "WeaveCallback",
    ],
    "dpo_config": [
        "DPOConfig",
        "FDivergenceConstants",  # deprecated import
        "FDivergenceType",  # deprecated import
    ],
    "dpo_trainer": ["DPOTrainer"],
    "grpo_config": ["GRPOConfig"],
    "grpo_trainer": ["GRPOTrainer"],
    "kto_config": ["KTOConfig"],
    "kto_trainer": ["KTOTrainer"],
    "model_config": ["ModelConfig"],
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
}

if TYPE_CHECKING:
    from .callbacks import (
        BEMACallback,
        LogCompletionsCallback,
        RichProgressCallback,
        SyncRefModelCallback,
        WeaveCallback,
    )
    from .dpo_config import (
        DPOConfig,
        FDivergenceConstants,  # deprecated import
        FDivergenceType,  # deprecated import
    )
    from .dpo_trainer import DPOTrainer
    from .grpo_config import GRPOConfig
    from .grpo_trainer import GRPOTrainer
    from .kto_config import KTOConfig
    from .kto_trainer import KTOTrainer
    from .model_config import ModelConfig
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
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
