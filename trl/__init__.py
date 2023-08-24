# flake8: noqa

__version__ = "0.6.0"

from .core import set_seed
from .extras import BestOfNSampler
from .import_utils import is_diffusers_available, is_peft_available
from .models import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PreTrainedModelWrapper,
    create_reference_model,
)
from .trainer import DataCollatorForCompletionOnlyLM, DPOTrainer, PPOConfig, PPOTrainer, RewardTrainer, SFTTrainer


if is_diffusers_available():
    from .models import (
        DDPOPipelineOutput,
        DDPOSchedulerOutput,
        DDPOStableDiffusionPipeline,
        DefaultDDPOStableDiffusionPipeline,
    )
    from .trainer import DDPOConfig, DDPOTrainer
