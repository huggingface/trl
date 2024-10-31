# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

__version__ = "0.12.0.dev0"

from typing import TYPE_CHECKING

from .import_utils import OptionalDependencyNotAvailable, _LazyModule, is_diffusers_available


_import_structure = {
    "commands.cli_utils": ["DPOScriptArguments", "SFTScriptArguments", "TrlParser", "init_zero_verbose"],
    "core": ["set_seed"],
    "data_utils": [
        "apply_chat_template",
        "extract_prompt",
        "is_conversational",
        "maybe_apply_chat_template",
        "maybe_extract_prompt",
        "maybe_unpair_preference_dataset",
        "unpair_preference_dataset",
    ],
    "environment": ["TextEnvironment", "TextHistory"],
    "extras": ["BestOfNSampler"],
    "import_utils": [
        "is_deepspeed_available",
        "is_diffusers_available",
        "is_llm_blender_available",
    ],
    "models": [
        "SUPPORTED_ARCHITECTURES",
        "AutoModelForCausalLMWithValueHead",
        "AutoModelForSeq2SeqLMWithValueHead",
        "PreTrainedModelWrapper",
        "create_reference_model",
        "setup_chat_format",
    ],
    "trainer": [
        "AlignPropConfig",
        "AlignPropTrainer",
        "BaseJudge",
        "BasePairwiseJudge",
        "BaseRankJudge",
        "BCOConfig",
        "BCOTrainer",
        "CPOConfig",
        "CPOTrainer",
        "DataCollatorForCompletionOnlyLM",
        "DPOConfig",
        "DPOTrainer",
        "FDivergenceConstants",
        "FDivergenceType",
        "GKDConfig",
        "GKDTrainer",
        "HfPairwiseJudge",
        "IterativeSFTTrainer",
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
        "PPOv2Config",
        "PPOv2Trainer",
        "RandomPairwiseJudge",
        "RandomRankJudge",
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
    "trainer.callbacks": ["RichProgressCallback", "SyncRefModelCallback"],
    "trainer.utils": ["get_kbit_device_map", "get_peft_config", "get_quantization_config"],
    "utils": ["ScriptArguments"],
}

try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["models"].extend(
        [
            "DDPOPipelineOutput",
            "DDPOSchedulerOutput",
            "DDPOStableDiffusionPipeline",
            "DefaultDDPOStableDiffusionPipeline",
        ]
    )
    _import_structure["trainer"].extend(["DDPOConfig", "DDPOTrainer"])

if TYPE_CHECKING:
    from .commands.cli_utils import DPOScriptArguments, SFTScriptArguments, TrlParser, init_zero_verbose
    from .core import set_seed
    from .data_utils import (
        apply_chat_template,
        extract_prompt,
        is_conversational,
        maybe_apply_chat_template,
        maybe_extract_prompt,
        maybe_unpair_preference_dataset,
        unpair_preference_dataset,
    )
    from .environment import TextEnvironment, TextHistory
    from .extras import BestOfNSampler
    from .import_utils import is_deepspeed_available, is_diffusers_available, is_llm_blender_available
    from .models import (
        SUPPORTED_ARCHITECTURES,
        AutoModelForCausalLMWithValueHead,
        AutoModelForSeq2SeqLMWithValueHead,
        PreTrainedModelWrapper,
        create_reference_model,
        setup_chat_format,
    )
    from .trainer import (
        AlignPropConfig,
        AlignPropTrainer,
        BaseJudge,
        BasePairwiseJudge,
        BaseRankJudge,
        BCOConfig,
        BCOTrainer,
        CPOConfig,
        CPOTrainer,
        DataCollatorForCompletionOnlyLM,
        DPOConfig,
        DPOTrainer,
        FDivergenceConstants,
        FDivergenceType,
        GKDConfig,
        GKDTrainer,
        HfPairwiseJudge,
        IterativeSFTTrainer,
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
        PPOv2Config,
        PPOv2Trainer,
        RandomPairwiseJudge,
        RandomRankJudge,
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
    from .trainer.callbacks import RichProgressCallback, SyncRefModelCallback
    from .trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config
    from .utils import ScriptArguments

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .models import (
            DDPOPipelineOutput,
            DDPOSchedulerOutput,
            DDPOStableDiffusionPipeline,
            DefaultDDPOStableDiffusionPipeline,
        )
        from .trainer import DDPOConfig, DDPOTrainer

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
