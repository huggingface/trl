import warnings
from dataclasses import dataclass, field
from typing import Optional

from ..core import flatten_dict
from ..import_utils import is_bitsandbytes_available, is_torchvision_available


@dataclass
class DDPOConfig(object):
    """
    Configuration class for DDPOTrainer
    """

    run_name: Optional[str] = field(
        default="",
        metadata={"help": "Run name for wandb logging and checkpoint saving."},
    )
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed for reproducibility."})
    logdir: Optional[str] = field(
        default="logs",
        metadata={"help": "Top-level logging directory for checkpoint saving."},
    )
    log_with: Optional[str] = field(
        default=None,
        metadata={
            "help": "Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"
        },
    )
    tracker_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the tracker (e.g. wandb_project)"},
    )
    accelerator_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator"},
    )
    project_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator project config (e.g. `logging_dir`)"},
    )
    tracker_project_name: Optional[str] = field(
        default="trl", metadata={"help": "Name of project to use for tracking"}
    )
    num_epochs: Optional[int] = field(default=100, metadata={"help": "Number of epochs to train."})
    save_freq: Optional[int] = field(
        default=1,
        metadata={"help": "Number of epochs between saving model checkpoints."},
    )
    num_checkpoint_limit: Optional[int] = field(
        default=5,
        metadata={"help": "Number of checkpoints to keep before overwriting old ones."},
    )
    mixed_precision: Optional[str] = field(default="fp16", metadata={"help": "Mixed precision training."})
    allow_tf32: Optional[bool] = field(default=True, metadata={"help": "Allow tf32 on Ampere GPUs."})
    resume_from: Optional[str] = field(default="", metadata={"help": "Resume training from a checkpoint."})
    sample_num_steps: Optional[int] = field(default=50, metadata={"help": "Number of sampler inference steps."})
    sample_eta: Optional[float] = field(default=1.0, metadata={"help": "Eta parameter for the DDIM sampler."})
    sample_guidance_scale: Optional[float] = field(default=5.0, metadata={"help": "Classifier-free guidance weight."})
    sample_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Batch size (per GPU!) to use for sampling."}
    )
    sample_num_batches_per_epoch: Optional[int] = field(
        default=2, metadata={"help": "Number of batches to sample per epoch."}
    )
    train_batch_size: Optional[int] = field(default=1, metadata={"help": "Batch size (per GPU!) to use for training."})
    train_use_8bit_adam: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the 8bit Adam optimizer from bitsandbytes."},
    )
    train_learning_rate: Optional[float] = field(default=3e-4, metadata={"help": "Learning rate."})
    train_adam_beta1: Optional[float] = field(default=0.9, metadata={"help": "Adam beta1."})
    train_adam_beta2: Optional[float] = field(default=0.999, metadata={"help": "Adam beta2."})
    train_adam_weight_decay: Optional[float] = field(default=1e-4, metadata={"help": "Adam weight decay."})
    train_adam_epsilon: Optional[float] = field(default=1e-8, metadata={"help": "Adam epsilon."})
    train_gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "Number of gradient accumulation steps."}
    )
    train_max_grad_norm: Optional[float] = field(
        default=1.0, metadata={"help": "Maximum gradient norm for gradient clipping."}
    )
    train_num_inner_epochs: Optional[int] = field(
        default=1, metadata={"help": "Number of inner epochs per outer epoch."}
    )
    train_cfg: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use classifier-free guidance during training."},
    )
    train_adv_clip_max: Optional[float] = field(default=5, metadata={"help": "Clip advantages to the range."})
    train_clip_range: Optional[float] = field(default=1e-4, metadata={"help": "The PPO clip range."})
    train_timestep_fraction: Optional[float] = field(
        default=1.0, metadata={"help": "The fraction of timesteps to train on."}
    )

    per_prompt_stat_tracking: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to track statistics for each prompt separately."},
    )

    per_prompt_stat_tracking_buffer_size: Optional[int] = field(
        default=16,
        metadata={"help": "Number of reward values to store in the buffer for each prompt."},
    )
    per_prompt_stat_tracking_min_count: Optional[int] = field(
        default=16,
        metadata={"help": "The minimum number of reward values to store in the buffer."},
    )
    async_reward_computation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to compute rewards asynchronously."},
    )
    max_workers: Optional[int] = field(
        default=2,
        metadata={"help": "The maximum number of workers to use for async reward computation."},
    )
    negative_prompts: Optional[str] = field(
        default="",
        metadata={"help": "Comma-separated list of prompts to use as negative examples."},
    )

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.log_with not in ["wandb", "tensorboard"]:
            warnings.warn(
                ("Accelerator tracking only supports image logging if `log_with` is set to 'wandb' or 'tensorboard'.")
            )

        if self.log_with == "wandb" and not is_torchvision_available():
            warnings.warn("Wandb image logging requires torchvision to be installed")

        if self.train_use_8bit_adam and not is_bitsandbytes_available():
            raise ImportError(
                "You need to install bitsandbytes to use 8bit Adam. "
                "You can install it with `pip install bitsandbytes`."
            )
