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

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import is_bitsandbytes_available

from ..core import flatten_dict


@dataclass
class DDPOConfig:
    r"""
    Configuration class for the [`DDPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(sys.argv[0])[: -len(".py")]`):
            Name of this experiment (by default is the file name without the extension name).
        run_name (`str`, *optional*, defaults to `""`):
            Name of this run.
        seed (`int`, *optional*, defaults to `0`):
            Random seed.
        log_with (`Literal["wandb", "tensorboard"]]` or `None`, *optional*, defaults to `None`):
            Log with either 'wandb' or 'tensorboard', check
            https://huggingface.co/docs/accelerate/usage_guides/tracking for more details.
        tracker_kwargs (`Dict`, *optional*, defaults to `{}`):
            Keyword arguments for the tracker (e.g. wandb_project).
        accelerator_kwargs (`Dict`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator.
        project_kwargs (`Dict`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator project config (e.g. `logging_dir`).
        tracker_project_name (`str`, *optional*, defaults to `"trl"`):
            Name of project to use for tracking.
        logdir (`str`, *optional*, defaults to `"logs"`):
            Top-level logging directory for checkpoint saving.
        num_epochs (`int`, *optional*, defaults to `100`):
            Number of epochs to train.
        save_freq (`int`, *optional*, defaults to `1`):
            Number of epochs between saving model checkpoints.
        num_checkpoint_limit (`int`, *optional*, defaults to `5`):
            Number of checkpoints to keep before overwriting old ones.
        mixed_precision (`str`, *optional*, defaults to `"fp16"`):
            Mixed precision training.
        allow_tf32 (`bool`, *optional*, defaults to `True`):
            Allow `tf32` on Ampere GPUs.
        resume_from (`str`, *optional*, defaults to `""`):
            Resume training from a checkpoint.
        sample_num_steps (`int`, *optional*, defaults to `50`):
            Number of sampler inference steps.
        sample_eta (`float`, *optional*, defaults to `1.0`):
            Eta parameter for the DDIM sampler.
        sample_guidance_scale (`float`, *optional*, defaults to `5.0`):
            Classifier-free guidance weight.
        sample_batch_size (`int`, *optional*, defaults to `1`):
            Batch size (per GPU) to use for sampling.
        sample_num_batches_per_epoch (`int`, *optional*, defaults to `2`):
            Number of batches to sample per epoch.
        train_batch_size (`int`, *optional*, defaults to `1`):
            Batch size (per GPU) to use for training.
        train_use_8bit_adam (`bool`, *optional*, defaults to `False`):
            Use 8bit Adam optimizer from bitsandbytes.
        train_learning_rate (`float`, *optional*, defaults to `3e-4`):
            Learning rate.
        train_adam_beta1 (`float`, *optional*, defaults to `0.9`):
            Adam beta1.
        train_adam_beta2 (`float`, *optional*, defaults to `0.999`):
            Adam beta2.
        train_adam_weight_decay (`float`, *optional*, defaults to `1e-4`):
            Adam weight decay.
        train_adam_epsilon (`float`, *optional*, defaults to `1e-8`):
            Adam epsilon.
        train_gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of gradient accumulation steps.
        train_max_grad_norm (`float`, *optional*, defaults to `1.0`):
            Maximum gradient norm for gradient clipping.
        train_num_inner_epochs (`int`, *optional*, defaults to `1`):
            Number of inner epochs per outer epoch.
        train_cfg (`bool`, *optional*, defaults to `True`):
            Whether to use classifier-free guidance during training.
        train_adv_clip_max (`float`, *optional*, defaults to `5.0`):
            Clip advantages to the range.
        train_clip_range (`float`, *optional*, defaults to `1e-4`):
            PPO clip range.
        train_timestep_fraction (`float`, *optional*, defaults to `1.0`):
            Fraction of timesteps to train on.
        per_prompt_stat_tracking (`bool`, *optional*, defaults to `False`):
            Whether to track statistics for each prompt separately.
        per_prompt_stat_tracking_buffer_size (`int`, *optional*, defaults to `16`):
            Number of reward values to store in the buffer for each prompt.
        per_prompt_stat_tracking_min_count (`int`, *optional*, defaults to `16`):
            Minimum number of reward values to store in the buffer.
        async_reward_computation (`bool`, *optional*, defaults to `False`):
            Whether to compute rewards asynchronously.
        max_workers (`int`, *optional*, defaults to `2`):
            Maximum number of workers to use for async reward computation.
        negative_prompts (`str`, *optional*, defaults to `""`):
            Comma-separated list of prompts to use as negative examples.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the final model checkpoint to the Hub.
    """

    exp_name: str = field(
        default=os.path.basename(sys.argv[0])[: -len(".py")],
        metadata={"help": "Name of this experiment (by default is the file name without the extension name)."},
    )
    run_name: str = field(
        default="",
        metadata={"help": "Name of this run."},
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed."},
    )
    log_with: Optional[str] = field(
        default=None,
        metadata={
            "help": "Log with either 'wandb' or 'tensorboard'.",
            "choices": ["wandb", "tensorboard"],
        },
    )
    tracker_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the tracker (e.g. wandb_project)."},
    )
    accelerator_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator."},
    )
    project_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator project config (e.g. `logging_dir`)."},
    )
    tracker_project_name: str = field(
        default="trl",
        metadata={"help": "Name of project to use for tracking."},
    )
    logdir: str = field(
        default="logs",
        metadata={"help": "Top-level logging directory for checkpoint saving."},
    )
    num_epochs: int = field(
        default=100,
        metadata={"help": "Number of epochs to train."},
    )
    save_freq: int = field(
        default=1,
        metadata={"help": "Number of epochs between saving model checkpoints."},
    )
    num_checkpoint_limit: int = field(
        default=5,
        metadata={"help": "Number of checkpoints to keep before overwriting old ones."},
    )
    mixed_precision: str = field(
        default="fp16",
        metadata={"help": "Mixed precision training."},
    )
    allow_tf32: bool = field(
        default=True,
        metadata={"help": "Allow `tf32` on Ampere GPUs."},
    )
    resume_from: str = field(
        default="",
        metadata={"help": "Resume training from a checkpoint."},
    )
    sample_num_steps: int = field(
        default=50,
        metadata={"help": "Number of sampler inference steps."},
    )
    sample_eta: float = field(
        default=1.0,
        metadata={"help": "Eta parameter for the DDIM sampler."},
    )
    sample_guidance_scale: float = field(
        default=5.0,
        metadata={"help": "Classifier-free guidance weight."},
    )
    sample_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size (per GPU) to use for sampling."},
    )
    sample_num_batches_per_epoch: int = field(
        default=2,
        metadata={"help": "Number of batches to sample per epoch."},
    )
    train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size (per GPU) to use for training."},
    )
    train_use_8bit_adam: bool = field(
        default=False,
        metadata={"help": "Use 8bit Adam optimizer from bitsandbytes."},
    )
    train_learning_rate: float = field(
        default=3e-4,
        metadata={"help": "Learning rate."},
    )
    train_adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Adam beta1."},
    )
    train_adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Adam beta2."},
    )
    train_adam_weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Adam weight decay."},
    )
    train_adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam epsilon."},
    )
    train_gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps."},
    )
    train_max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for gradient clipping."},
    )
    train_num_inner_epochs: int = field(
        default=1,
        metadata={"help": "Number of inner epochs per outer epoch."},
    )
    train_cfg: bool = field(
        default=True,
        metadata={"help": "Whether to use classifier-free guidance during training."},
    )
    train_adv_clip_max: float = field(
        default=5.0,
        metadata={"help": "Clip advantages to the range."},
    )
    train_clip_range: float = field(
        default=1e-4,
        metadata={"help": "PPO clip range."},
    )
    train_timestep_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of timesteps to train on."},
    )
    per_prompt_stat_tracking: bool = field(
        default=False,
        metadata={"help": "Whether to track statistics for each prompt separately."},
    )
    per_prompt_stat_tracking_buffer_size: int = field(
        default=16,
        metadata={"help": "Number of reward values to store in the buffer for each prompt."},
    )
    per_prompt_stat_tracking_min_count: int = field(
        default=16,
        metadata={"help": "Minimum number of reward values to store in the buffer."},
    )
    async_reward_computation: bool = field(
        default=False,
        metadata={"help": "Whether to compute rewards asynchronously."},
    )
    max_workers: int = field(
        default=2,
        metadata={"help": "Maximum number of workers to use for async reward computation."},
    )
    negative_prompts: str = field(
        default="",
        metadata={"help": "Comma-separated list of prompts to use as negative examples."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the final model checkpoint to the Hub."},
    )

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.train_use_8bit_adam and not is_bitsandbytes_available():
            raise ImportError(
                "You need to install bitsandbytes to use 8bit Adam. "
                "You can install it with `pip install bitsandbytes`."
            )
