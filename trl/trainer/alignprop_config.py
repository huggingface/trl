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
from typing import Any, Optional

from transformers import is_bitsandbytes_available

from ..core import flatten_dict


@dataclass
class AlignPropConfig:
    r"""
    Configuration class for the [`AlignPropTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(sys.argv[0])[: -len(".py")]`):
            Name of this experiment (defaults to the file name without the extension).
        run_name (`str`, *optional*, defaults to `""`):
            Name of this run.
        seed (`int`, *optional*, defaults to `0`):
            Random seed for reproducibility.
        log_with (`str` or `None`, *optional*, defaults to `None`):
            Log with either `"wandb"` or `"tensorboard"`. Check
            [tracking](https://huggingface.co/docs/accelerate/usage_guides/tracking) for more details.
        log_image_freq (`int`, *optional*, defaults to `1`):
            Frequency for logging images.
        tracker_kwargs (`dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the tracker (e.g., `wandb_project`).
        accelerator_kwargs (`dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator.
        project_kwargs (`dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator project config (e.g., `logging_dir`).
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
            Path to resume training from a checkpoint.
        sample_num_steps (`int`, *optional*, defaults to `50`):
            Number of sampler inference steps.
        sample_eta (`float`, *optional*, defaults to `1.0`):
            Eta parameter for the DDIM sampler.
        sample_guidance_scale (`float`, *optional*, defaults to `5.0`):
            Classifier-free guidance weight.
        train_batch_size (`int`, *optional*, defaults to `1`):
            Batch size for training.
        train_use_8bit_adam (`bool`, *optional*, defaults to `False`):
            Whether to use the 8bit Adam optimizer from `bitsandbytes`.
        train_learning_rate (`float`, *optional*, defaults to `1e-3`):
            Learning rate.
        train_adam_beta1 (`float`, *optional*, defaults to `0.9`):
            Beta1 for Adam optimizer.
        train_adam_beta2 (`float`, *optional*, defaults to `0.999`):
            Beta2 for Adam optimizer.
        train_adam_weight_decay (`float`, *optional*, defaults to `1e-4`):
            Weight decay for Adam optimizer.
        train_adam_epsilon (`float`, *optional*, defaults to `1e-8`):
            Epsilon value for Adam optimizer.
        train_gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of gradient accumulation steps.
        train_max_grad_norm (`float`, *optional*, defaults to `1.0`):
            Maximum gradient norm for gradient clipping.
        negative_prompts (`str` or `None`, *optional*, defaults to `None`):
            Comma-separated list of prompts to use as negative examples.
        truncated_backprop_rand (`bool`, *optional*, defaults to `True`):
            If `True`, randomized truncation to different diffusion timesteps is used.
        truncated_backprop_timestep (`int`, *optional*, defaults to `49`):
            Absolute timestep to which the gradients are backpropagated. Used only if `truncated_backprop_rand=False`.
        truncated_rand_backprop_minmax (`tuple[int, int]`, *optional*, defaults to `(0, 50)`):
            Range of diffusion timesteps for randomized truncated backpropagation.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the final model to the Hub.
    """

    exp_name: str = field(
        default=os.path.basename(sys.argv[0])[: -len(".py")],
        metadata={"help": "Name of this experiment (defaults to the file name without the extension)."},
    )
    run_name: str = field(default="", metadata={"help": "Name of this run."})
    seed: int = field(default=0, metadata={"help": "Random seed for reproducibility."})
    log_with: Optional[str] = field(
        default=None,
        metadata={"help": "Log with either 'wandb' or 'tensorboard'.", "choices": ["wandb", "tensorboard"]},
    )
    log_image_freq: int = field(default=1, metadata={"help": "Frequency for logging images."})
    tracker_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the tracker (e.g., `wandb_project`)."},
    )
    accelerator_kwargs: dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Keyword arguments for the accelerator."}
    )
    project_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator project config (e.g., `logging_dir`)."},
    )
    tracker_project_name: str = field(default="trl", metadata={"help": "Name of project to use for tracking."})
    logdir: str = field(default="logs", metadata={"help": "Top-level logging directory for checkpoint saving."})
    num_epochs: int = field(default=100, metadata={"help": "Number of epochs to train."})
    save_freq: int = field(default=1, metadata={"help": "Number of epochs between saving model checkpoints."})
    num_checkpoint_limit: int = field(
        default=5, metadata={"help": "Number of checkpoints to keep before overwriting old ones."}
    )
    mixed_precision: str = field(
        default="fp16",
        metadata={
            "help": "Mixed precision training. Possible values are 'fp16', 'bf16', 'none'.",
            "choices": ["fp16", "bf16", "none"],
        },
    )
    allow_tf32: bool = field(default=True, metadata={"help": "Allow `tf32` on Ampere GPUs."})
    resume_from: str = field(default="", metadata={"help": "Path to resume training from a checkpoint."})
    sample_num_steps: int = field(default=50, metadata={"help": "Number of sampler inference steps."})
    sample_eta: float = field(default=1.0, metadata={"help": "Eta parameter for the DDIM sampler."})
    sample_guidance_scale: float = field(default=5.0, metadata={"help": "Classifier-free guidance weight."})
    train_batch_size: int = field(default=1, metadata={"help": "Batch size for training."})
    train_use_8bit_adam: bool = field(
        default=False, metadata={"help": "Whether to use the 8bit Adam optimizer from `bitsandbytes`."}
    )
    train_learning_rate: float = field(default=1e-3, metadata={"help": "Learning rate."})
    train_adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer."})
    train_adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer."})
    train_adam_weight_decay: float = field(default=1e-4, metadata={"help": "Weight decay for Adam optimizer."})
    train_adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon value for Adam optimizer."})
    train_gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of gradient accumulation steps."}
    )
    train_max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm for gradient clipping."})
    negative_prompts: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of prompts to use as negative examples."},
    )
    truncated_backprop_rand: bool = field(
        default=True,
        metadata={"help": "If `True`, randomized truncation to different diffusion timesteps is used."},
    )
    truncated_backprop_timestep: int = field(
        default=49,
        metadata={
            "help": "Absolute timestep to which the gradients are backpropagated. Used only if "
            "`truncated_backprop_rand=False`."
        },
    )
    truncated_rand_backprop_minmax: tuple[int, int] = field(
        default=(0, 50),
        metadata={
            "help": "Range of diffusion timesteps for randomized truncated backpropagation.",
        },
    )
    push_to_hub: bool = field(default=False, metadata={"help": "Whether to push the final model to the Hub."})

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
