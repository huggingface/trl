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

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from . import _compat
from .import_utils import _LazyModule


try:
    __version__ = version("trl")
except PackageNotFoundError:
    __version__ = "unknown"

_import_structure = {
    "chat_template_utils": ["add_response_schema", "clone_chat_template", "get_training_chat_template"],
    "data_utils": [
        "apply_chat_template",
        "extract_prompt",
        "is_conversational",
        "is_conversational_from_value",
        "maybe_apply_chat_template",
        "maybe_convert_to_chatml",
        "maybe_extract_prompt",
        "maybe_unpair_preference_dataset",
        "pack_dataset",
        "prepare_multimodal_messages",
        "prepare_multimodal_messages_vllm",
        "truncate_dataset",
        "unpair_preference_dataset",
    ],
    "models": [
        "create_reference_model",
    ],
    "scripts": ["DatasetMixtureConfig", "ScriptArguments", "TrlParser", "get_dataset", "init_zero_verbose"],
    "trainer": [
        "BEMACallback",
        "DPOConfig",
        "DPOTrainer",
        "FDivergenceConstants",  # deprecated import
        "FDivergenceType",  # deprecated import
        "GRPOConfig",
        "GRPOTrainer",
        "KTOConfig",
        "KTOTrainer",
        "LogCompletionsCallback",
        "ModelConfig",
        "RewardConfig",
        "RewardTrainer",
        "RichProgressCallback",
        "RLOOConfig",
        "RLOOTrainer",
        "SFTConfig",
        "SFTTrainer",
        "SyncRefModelCallback",
        "WeaveCallback",
        "get_kbit_device_map",
        "get_peft_config",
        "get_quantization_config",
    ],
}

if TYPE_CHECKING:
    from .chat_template_utils import add_response_schema, clone_chat_template, get_training_chat_template
    from .data_utils import (
        apply_chat_template,
        extract_prompt,
        is_conversational,
        is_conversational_from_value,
        maybe_apply_chat_template,
        maybe_convert_to_chatml,
        maybe_extract_prompt,
        maybe_unpair_preference_dataset,
        pack_dataset,
        prepare_multimodal_messages,
        prepare_multimodal_messages_vllm,
        truncate_dataset,
        unpair_preference_dataset,
    )
    from .models import (
        create_reference_model,
    )
    from .scripts import DatasetMixtureConfig, ScriptArguments, TrlParser, get_dataset, init_zero_verbose
    from .trainer import (
        BEMACallback,
        DPOConfig,
        DPOTrainer,
        FDivergenceConstants,  # deprecated import
        FDivergenceType,  # deprecated import
        GRPOConfig,
        GRPOTrainer,
        KTOConfig,
        KTOTrainer,
        LogCompletionsCallback,
        ModelConfig,
        RewardConfig,
        RewardTrainer,
        RichProgressCallback,
        RLOOConfig,
        RLOOTrainer,
        SFTConfig,
        SFTTrainer,
        SyncRefModelCallback,
        WeaveCallback,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
