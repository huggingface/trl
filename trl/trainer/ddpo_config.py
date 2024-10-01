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

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import is_bitsandbytes_available, is_torchvision_available

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
        log_with (`Optional[Literal["wandb", "tensorboard"]]`, *optional*, defaults to `None`):
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
            Whether or not to use classifier-free guidance during training.
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
        negative_prompts (`Optional[str]`, *optional*, defaults to `""`):
            Comma-separated list of prompts to use as negative examples.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the final model checkpoint to the Hub.
    """

    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    run_name: str = ""
    seed: int = 0
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    tracker_kwargs: dict = field(default_factory=dict)
    accelerator_kwargs: dict = field(default_factory=dict)
    project_kwargs: dict = field(default_factory=dict)
    tracker_project_name: str = "trl"
    logdir: str = "logs"
    num_epochs: int = 100
    save_freq: int = 1
    num_checkpoint_limit: int = 5
    mixed_precision: str = "fp16"
    allow_tf32: bool = True
    resume_from: str = ""
    sample_num_steps: int = 50
    sample_eta: float = 1.0
    sample_guidance_scale: float = 5.0
    sample_batch_size: int = 1
    sample_num_batches_per_epoch: int = 2
    train_batch_size: int = 1
    train_use_8bit_adam: bool = False
    train_learning_rate: float = 3e-4
    train_adam_beta1: float = 0.9
    train_adam_beta2: float = 0.999
    train_adam_weight_decay: float = 1e-4
    train_adam_epsilon: float = 1e-8
    train_gradient_accumulation_steps: int = 1
    train_max_grad_norm: float = 1.0
    train_num_inner_epochs: int = 1
    train_cfg: bool = True
    train_adv_clip_max: float = 5.0
    train_clip_range: float = 1e-4
    train_timestep_fraction: float = 1.0
    per_prompt_stat_tracking: bool = False
    per_prompt_stat_tracking_buffer_size: int = 16
    per_prompt_stat_tracking_min_count: int = 16
    async_reward_computation: bool = False
    max_workers: int = 2
    negative_prompts: str = ""
    push_to_hub: bool = False

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.log_with not in ["wandb", "tensorboard"]:
            warnings.warn(
                "Accelerator tracking only supports image logging if `log_with` is set to 'wandb' or 'tensorboard'."
            )

        if self.log_with == "wandb" and not is_torchvision_available():
            warnings.warn("Wandb image logging requires torchvision to be installed")

        if self.train_use_8bit_adam and not is_bitsandbytes_available():
            raise ImportError(
                "You need to install bitsandbytes to use 8bit Adam. "
                "You can install it with `pip install bitsandbytes`."
            )
