# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from .import_utils import _LazyModule


try:
    __version__ = version("trl")
except PackageNotFoundError:
    __version__ = "unknown"

_import_structure = {
    "scripts": ["DatasetMixtureConfig", "ScriptArguments", "TrlParser", "get_dataset", "init_zero_verbose"],
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
        "SUPPORTED_ARCHITECTURES",
        "AutoModelForCausalLMWithValueHead",
        "AutoModelForSeq2SeqLMWithValueHead",
        "PreTrainedModelWrapper",
        "clone_chat_template",
        "create_reference_model",
    ],
    "trainer": [
        "AllTrueJudge",
        "BaseBinaryJudge",
        "BaseJudge",
        "BasePairwiseJudge",
        "BaseRankJudge",
        "BCOConfig",
        "BCOTrainer",
        "CPOConfig",
        "CPOTrainer",
        "DPOConfig",
        "DPOTrainer",
        "FDivergenceConstants",
        "FDivergenceType",
        "GKDConfig",
        "GKDTrainer",
        "GRPOConfig",
        "GRPOTrainer",
        "HfPairwiseJudge",
        "KTOConfig",
        "KTOTrainer",
        "LogCompletionsCallback",
        "ModelConfig",
        "NashMDConfig",
        "NashMDTrainer",
        "OnlineDPOConfig",
        "OnlineDPOTrainer",
        "OpenAIPairwiseJudge",
        "ORPOConfig",
        "ORPOTrainer",
        "PairRMJudge",
        "PPOConfig",
        "PPOTrainer",
        "PRMConfig",
        "PRMTrainer",
        "RewardConfig",
        "RewardTrainer",
        "RLOOConfig",
        "RLOOTrainer",
        "SFTConfig",
        "SFTTrainer",
        "WinRateCallback",
        "XPOConfig",
        "XPOTrainer",
    ],
    "trainer.callbacks": [
        "BEMACallback",
        "MergeModelCallback",
        "RichProgressCallback",
        "SyncRefModelCallback",
        "WeaveCallback",
    ],
    "trainer.utils": ["get_kbit_device_map", "get_peft_config", "get_quantization_config"],
}

if TYPE_CHECKING:
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
        SUPPORTED_ARCHITECTURES,
        AutoModelForCausalLMWithValueHead,
        AutoModelForSeq2SeqLMWithValueHead,
        PreTrainedModelWrapper,
        clone_chat_template,
        create_reference_model,
    )
    from .scripts import DatasetMixtureConfig, ScriptArguments, TrlParser, get_dataset, init_zero_verbose
    from .trainer import (
        AllTrueJudge,
        BaseBinaryJudge,
        BaseJudge,
        BasePairwiseJudge,
        BaseRankJudge,
        BCOConfig,
        BCOTrainer,
        CPOConfig,
        CPOTrainer,
        DPOConfig,
        DPOTrainer,
        FDivergenceConstants,
        FDivergenceType,
        GKDConfig,
        GKDTrainer,
        GRPOConfig,
        GRPOTrainer,
        HfPairwiseJudge,
        KTOConfig,
        KTOTrainer,
        LogCompletionsCallback,
        ModelConfig,
        NashMDConfig,
        NashMDTrainer,
        OnlineDPOConfig,
        OnlineDPOTrainer,
        OpenAIPairwiseJudge,
        ORPOConfig,
        ORPOTrainer,
        PairRMJudge,
        PPOConfig,
        PPOTrainer,
        PRMConfig,
        PRMTrainer,
        RewardConfig,
        RewardTrainer,
        RLOOConfig,
        RLOOTrainer,
        SFTConfig,
        SFTTrainer,
        WinRateCallback,
        XPOConfig,
        XPOTrainer,
    )
    from .trainer.callbacks import (
        BEMACallback,
        MergeModelCallback,
        RichProgressCallback,
        SyncRefModelCallback,
        WeaveCallback,
    )
    from .trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


# Monkey-patch for vLLM.
# Bug introduced in https://github.com/vllm-project/vllm/pull/52
# Fixed inhttps://github.com/vllm-project/vllm/pull/28471 (released in v0.11.1)
# Since TRL currently only supports vLLM v0.10.2, we patch it here. This can be removed when TRL requires vLLM >=0.11.1
from .import_utils import is_vllm_available  # noqa: E402


if is_vllm_available():
    import os

    os.environ["VLLM_LOGGING_LEVEL"] = os.getenv("VLLM_LOGGING_LEVEL", "ERROR")
    import vllm.model_executor.model_loader.weight_utils
    from tqdm import tqdm

    class DisabledTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    # overwrite the class in the dependency
    vllm.model_executor.model_loader.weight_utils.DisabledTqdm = DisabledTqdm
