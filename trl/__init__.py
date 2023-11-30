# flake8: noqa

__version__ = "0.7.5.dev0"

from .core import set_seed
from .environment import TextEnvironment, TextHistory
from .extras import BestOfNSampler
from .import_utils import is_diffusers_available, is_peft_available, is_wandb_available, is_xpu_available
from .models import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PreTrainedModelWrapper,
    create_reference_model,
)
from .trainer import (
    DataCollatorForCompletionOnlyLM,
    DPOTrainer,
    IterativeSFTTrainer,
    PPOConfig,
    PPOTrainer,
    RewardConfig,
    RewardTrainer,
    SFTTrainer,
)


if is_diffusers_available():
    from .models import (
        DDPOPipelineOutput,
        DDPOSchedulerOutput,
        DDPOStableDiffusionPipeline,
        DefaultDDPOStableDiffusionPipeline,
    )
    from .trainer import DDPOConfig, DDPOTrainer
