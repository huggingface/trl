# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import flatten_dict


@dataclass
class PPOConfig(object):
    """
    Configuration class for PPOTrainer

    Args:
        model_name (`str`, *optional*, defaults to `None`):
            Name of model to use - used only for tracking purposes
        steps (`int`, *optional*, defaults to 20000):
            Number of training steps
        learning_rate (`float`, *optional*, defaults to 1.41e-5):
            Adam learning rate
        adap_kl_ctrl (`bool`, *optional*, defaults to True):
            Use adaptive KL control, otherwise linear
        init_kl_coef (`float`, *optional*, defaults to 0.2):
            Initial KL penalty coefficient (used for adaptive and linear control)
        target (`float`, *optional*, defaults to 6):
            Target KL value for adaptive KL control
        horizon (`float`, *optional*, defaults to 10000):
            Horizon for adaptive KL control
        gamma (`float`, *optional*, defaults to 1):
            Gamma parameter for advantage calculation
        lam (`float`, *optional*, defaults to 0.95):
            Lambda parameter for advantage calculation
        cliprange (`float`, *optional*, defaults to 0.2):
            Range for clipping in PPO policy gradient loss
        cliprange_value (`float`, *optional*, defaults to 0.2):
            Range for clipping values in loss calculation
        vf_coef (`float`, *optional*, defaults to 0.1):
            Scaling factor for value loss
        batch_size (`int`, *optional*, defaults to 256):
            Number of samples per optimisation step
        forward_batch_size (`int`, *optional*, defaults to 16):
            Number of samples forward passed through model at a time
        mini_batch_size (`int`, *optional*, defaults to 1):
            Number of samples optimized inside PPO together
        ppo_epochs (`int`, *optional*, defaults to 4):
            Number of optimisation epochs per batch of samples
        remove_unused_columns (`bool`, *optional*, defaults to True):
            Remove unused columns from the dataset if `datasets.Dataset` is used
        log_with (`str`, *optional*, defaults to `None`):
            Log with either "wandb" or "tensorboard", check
            https://huggingface.co/docs/accelerate/usage_guides/tracking for more details
        accelerator_kwargs (`dict`, *optional*, defaults to {}):
            Keyword arguments for the accelerator (e.g. `logging_dir`)
        tracker_kwargs (`dict`, *optional*, defaults to {}):
            Keyword arguments for the tracker (e.g. wandb_project)
        tracker_project_name (`str`, *optional*, defaults to "trl"):
            Name of project to use for tracking
        max_grad_norm (`float`, *optional*, defaults to `None`):
            Maximum gradient norm for gradient clipping
        seed (`int`, *optional*, defaults to 0):
            Seed value for random generations
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        steps: Optional[int] = 20000,
        learning_rate: Optional[float] = 1e-5,
        adap_kl_ctrl: Optional[bool] = True,
        init_kl_coef: Optional[float] = 0.2,
        target: Optional[float] = 6,
        horizon: Optional[float] = 10000,
        gamma: Optional[float] = 1,
        lam: Optional[float] = 0.95,
        cliprange: Optional[float] = 0.2,
        cliprange_value: Optional[float] = 0.2,
        vf_coef: Optional[float] = 0.1,
        batch_size: Optional[int] = 256,
        forward_batch_size: Optional[int] = None,
        mini_batch_size: Optional[int] = 1,
        ppo_epochs: Optional[int] = 4,
        remove_unused_columns: Optional[bool] = True,
        log_with: Optional[str] = None,
        tracker_kwargs: Optional[dict] = {},
        accelerator_kwargs: Optional[dict] = {},
        tracker_project_name: Optional[str] = "trl",
        max_grad_norm: Optional[float] = None,
        seed: Optional[int] = 0,
    ):
        self.model_name = model_name
        self.steps = steps
        self.learning_rate = learning_rate
        self.adap_kl_ctrl = adap_kl_ctrl
        self.init_kl_coef = init_kl_coef
        self.target = target
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.cliprange = cliprange
        self.cliprange_value = cliprange_value
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        if forward_batch_size is not None:
            warnings.warn(
                "Note that using `forward_batch_size` is deprecated, use `mini_batch_size` instead. By setting it you overwrite `mini_batch_size` which affects both the batch size during forward passes and also the mini batch size for PPO optimization."
            )
            self.mini_batch_size = forward_batch_size
        else:
            self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.remove_unused_columns = remove_unused_columns
        self.seed = seed

        self.log_with = log_with
        # check if wandb is installed
        if self.log_with == "wandb":
            # raise error if wandb is not installed
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Please install wandb to use wandb logging. You can do this by running `pip install wandb`."
                )

        self.tracker_kwargs = tracker_kwargs
        self.accelerator_kwargs = accelerator_kwargs
        self.tracker_project_name = tracker_project_name
        self.max_grad_norm = max_grad_norm

        self.total_ppo_epochs = int(np.ceil(steps / batch_size))

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
