# flake8: noqa

__version__ = "0.9.6"

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
        "ModelConfig",
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
    ],
    "commands": [],
    "commands.cli_utils": ["init_zero_verbose", "SFTScriptArguments", "DPOScriptArguments", "TrlParser"],
    "trainer.utils": ["get_kbit_device_map", "get_peft_config", "get_quantization_config", "RichProgressCallback"],
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
    from .environment import TextEnvironment, TextHistory
    from .extras import BestOfNSampler
    from .import_utils import (
        is_bitsandbytes_available,
        is_diffusers_available,
        is_npu_available,
        is_peft_available,
        is_pil_available,
        is_wandb_available,
        is_xpu_available,
    )
    from .models import (
        AutoModelForCausalLMWithValueHead,
        AutoModelForSeq2SeqLMWithValueHead,
        PreTrainedModelWrapper,
        create_reference_model,
        setup_chat_format,
        SUPPORTED_ARCHITECTURES,
    )
    from .trainer import (
        DataCollatorForCompletionOnlyLM,
        DPOConfig,
        DPOTrainer,
        CPOConfig,
        CPOTrainer,
        AlignPropConfig,
        AlignPropTrainer,
        IterativeSFTTrainer,
        KTOConfig,
        KTOTrainer,
        ModelConfig,
        ORPOConfig,
        ORPOTrainer,
        PPOConfig,
        PPOTrainer,
        RewardConfig,
        RewardTrainer,
        SFTConfig,
        SFTTrainer,
        FDivergenceConstants,
        FDivergenceType,
    )
    from .trainer.utils import get_kbit_device_map, get_peft_config, get_quantization_config, RichProgressCallback
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
