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
from dataclasses import dataclass
from typing import Optional

@dataclass
class PPOConfig(object):
    """
    Configuration class for PPOTrainer

    Args:
        model_name (`str`, *optional*, defaults to `None`):
            Name of model to use - used only for tracking purposes
        use_length_sampler (`bool`, *optional*, defaults to True):
            Use the `LengthSampler` to sample the length of the input text
        txt_in_min_len (`int`, *optional*, defaults to 2):
            Minimum length of input text
        txt_in_max_len (`int`, *optional*, defaults to 8):
            Maximum length of input text
        txt_out_min_len (`int`, *optional*, defaults to 4):
            Minimum length of output text used by the `LengthSampler`
        txt_out_max_len (`int`, *optional*, defaults to 16):
            Maximum length of output text used by the `LengthSampler`
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
            Lambda parameter for advantage calcualation
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
        ppo_epochs (`int`, *optional*, defaults to 4):
            Number of optimisation epochs per batch of samples
        log_with_wandb (`bool`, *optional*, defaults to True):
            Log with wandb
        wandb_project (`str`, *optional*, defaults to "trl"):
            Name of wandb project
    """
    def __init__(
        self, 
        model_name: Optional[str] = None,
        use_length_sampler: Optional[bool] = True,
        txt_in_min_len: Optional[int] = 2,
        txt_in_max_len: Optional[int] = 8,
        txt_out_min_len: Optional[int] = 4,
        txt_out_max_len: Optional[int] = 16,
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
        forward_batch_size: Optional[int] = 16,
        ppo_epochs: Optional[int] = 4,
        log_with_wandb: Optional[bool] = True,
        wandb_project: Optional[str] = "trl",
    ):
        self.model_name = model_name
        self.use_length_sampler = use_length_sampler
        self.txt_in_min_len = txt_in_min_len
        self.txt_in_max_len = txt_in_max_len
        self.txt_out_min_len = txt_out_min_len
        self.txt_out_max_len = txt_out_max_len
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
        self.forward_batch_size = forward_batch_size
        self.ppo_epochs = ppo_epochs
        self.log_with_wandb = log_with_wandb
        self.wandb_project = wandb_project

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return output_dict