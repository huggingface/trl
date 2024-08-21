# flake8: noqa

__version__ = "0.9.6.dev0"

from typing import TYPE_CHECKING
from .import_utils import _LazyModule, is_diffusers_available, OptionalDependencyNotAvailable

_import_structure = {
    "core": [
        "set_seed",
    ],
    "environment": [
        "TextEnvironment",
        "TextHistory",
    ],
    "extras": [
        "BestOfNSampler",
    ],
    "import_utils": [
        "is_bitsandbytes_available",
        "is_diffusers_available",
        "is_npu_available",
        "is_peft_available",
        "is_pil_available",
        "is_wandb_available",
        "is_xpu_available",
        "is_llmblender_available",
        "is_openai_available",
    ],
    "models": [
        "AutoModelForCausalLMWithValueHead",
        "AutoModelForSeq2SeqLMWithValueHead",
        "PreTrainedModelWrapper",
        "create_reference_model",
        "setup_chat_format",
        "SUPPORTED_ARCHITECTURES",
    ],
    "trainer": [
        "DataCollatorForCompletionOnlyLM",
        "DPOConfig",
        "DPOTrainer",
        "CPOConfig",
        "CPOTrainer",
        "AlignPropConfig",
        "AlignPropTrainer",
        "IterativeSFTTrainer",
        "KTOConfig",
        "KTOTrainer",
        "BCOConfig",
        "BCOTrainer",
        "ModelConfig",
        "OnlineDPOConfig",
        "OnlineDPOTrainer",
        "GRPOConfig",
        "GRPOTrainer",
        "ORPOConfig",
        "ORPOTrainer",
        "PPOConfig",
        "PPOTrainer",
        "RewardConfig",
        "RewardTrainer",
        "SFTConfig",
        "SFTTrainer",
        "FDivergenceConstants",
        "FDivergenceType",
        "WinRateCallback",
        "BaseJudge",
        "BaseRankJudge",
        "BasePairwiseJudge",
        "RandomRankJudge",
        "RandomPairwiseJudge",
        "HfPairwiseJudge",
        "OpenAIPairwiseJudge",
    ],
    "commands": [],
    "commands.cli_utils": ["init_zero_verbose", "SFTScriptArguments", "DPOScriptArguments", "TrlParser"],
    "trainer.callbacks": ["RichProgressCallback", "SyncRefModelCallback"],
    "trainer.utils": ["get_kbit_device_map", "get_peft_config", "get_quantization_config"],
    "multitask_prompt_tuning": [
        "MultitaskPromptEmbedding",
        "MultitaskPromptTuningConfig",
        "MultitaskPromptTuningInit",
    ],
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
    from .core import set_seed
    from .extras import BestOfNSampler
    from .environment import TextEnvironment, TextHistory
    from .import_utils import (
        is_npu_available,
        is_pil_available,
        is_xpu_available,
        is_peft_available,
        is_wandb_available,
        is_openai_available,
        is_diffusers_available,
        is_llmblender_available,
        is_bitsandbytes_available,
    )
    from .models import (
        setup_chat_format,
        PreTrainedModelWrapper,
        create_reference_model,
        SUPPORTED_ARCHITECTURES,
        AutoModelForCausalLMWithValueHead,
        AutoModelForSeq2SeqLMWithValueHead,
    )
    from .trainer import (
        DPOConfig,
        CPOConfig,
        KTOConfig,
        BCOConfig,
        PPOConfig,
        SFTConfig,
        BaseJudge,
        DPOTrainer,
        CPOTrainer,
        KTOTrainer,
        BCOTrainer,
        ORPOConfig,
        PPOTrainer,
        SFTTrainer,
        GRPOConfig,
        ModelConfig,
        ORPOTrainer,
        GRPOTrainer,
        RewardConfig,
        RewardTrainer,
        BaseRankJudge,
        AlignPropConfig,
        OnlineDPOConfig,
        FDivergenceType,
        WinRateCallback,
        RandomRankJudge,
        HfPairwiseJudge,
        AlignPropTrainer,
        OnlineDPOTrainer,
        BasePairwiseJudge,
        IterativeSFTTrainer,
        RandomPairwiseJudge,
        OpenAIPairwiseJudge,
        FDivergenceConstants,
        DataCollatorForCompletionOnlyLM,
    )
    from .trainer.callbacks import RichProgressCallback, SyncRefModelCallback
    from .trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config
    from .commands.cli_utils import init_zero_verbose, SFTScriptArguments, DPOScriptArguments, TrlParser

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
