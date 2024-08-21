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
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import tyro
from typing_extensions import Annotated

from trl.trainer.utils import exact_div

from ..core import flatten_dict
from ..import_utils import is_wandb_available


JSONDict = Annotated[Optional[dict], tyro.conf.arg(metavar="JSON", constructor=json.loads)]


@dataclass
class GRPOConfig:
    """
    Configuration class for GRPOTrainer. It is eavily inspired by the original PPO work in the ppo_config.py file.
    """

    # common parameters
    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    """the name of this experiment (by default is the file name without the extension name)"""
    seed: int = 0
    """Seed value for random generations"""
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    """Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"""
    task_name: Optional[str] = None
    """Name of task to use - used only for tracking purposes"""
    model_name: Optional[str] = "gpt2"
    """Name of model to use - used only for tracking purposes"""
    query_dataset: Optional[str] = "imdb"
    """Name of dataset to query - used only for tracking purposes"""
    reward_model: Optional[str] = "sentiment-analysis:lvwerra/distilbert-imdb"
    """The reward model to use - used only for tracking purposes"""
    remove_unused_columns: bool = True
    """Remove unused columns from the dataset if `datasets.Dataset` is used"""
    tracker_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the tracker (e.g. python grpo.py --tracker_kwargs='{"wandb": {"entity": "my_wandb_entity", "name": "my_exp_name"}}'"""
    accelerator_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the accelerator"""
    project_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for the accelerator project config (e.g. `logging_dir`)"""
    tracker_project_name: str = "trl"
    """Name of project to use for tracking"""
    push_to_hub_if_best_kwargs: JSONDict = field(default_factory=dict)
    """Keyword arguments for pushing model to the hub during training (e.g. repo_id)"""

    # hyperparameters
    steps: int = 20000
    """Number of training steps"""
    learning_rate: float = 1.41e-5
    """Adam learning rate"""
    gamma: float = 1
    """Gamma parameter for advantage calculation"""
    lam: float = 0.95
    """Lambda parameter for advantage calculation"""
    cliprange: float = 0.2
    """Range for clipping in GRPO policy gradient loss"""
    cliprange_value: float = 0.2
    """Range for clipping values in loss calculation"""
    vf_coef: float = 0.1
    """Scaling factor for value loss"""
    batch_size: int = 128
    """Number of samples per optimisation step"""
    forward_batch_size: Optional[int] = None
    """DEPRECATED: use `mini_batch_size` instead, which does the same thing."""
    mini_batch_size: int = 128
    """Number of samples optimized in each mini batch"""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""
    world_size: tyro.conf.Suppress[int] = None
    """The world size for distributed training"""
    grpo_epochs: int = 4
    """Number of optimisation epochs per batch of samples"""
    max_grad_norm: Optional[float] = None
    """Maximum gradient norm for gradient clipping"""
    optimize_cuda_cache: Optional[bool] = None
    """DEPRECATED: use `optimize_device_cache` instead, which does the same thing."""
    optimize_device_cache: Optional[bool] = False
    """Optimize device cache for slightly more memory-efficient training"""
    early_stopping: bool = False
    """Whether to stop the GRPO optimization loop early is the KL too high"""
    compare_steps: int = 1
    """Number of steps between comparison of the current reward with the best seen so far"""
    ratio_threshold: float = 10.0
    """Skip mini-batches with high GRPO ratios that can cause loss spikes"""
    use_score_scaling: bool = False
    """Use score scaling"""
    use_score_norm: bool = False
    """Use score normalization. Only applicable if use_score_scaling is True"""
    score_clip: Optional[float] = None
    """Score clipping"""
    whiten_rewards: bool = False
    """Whiten the rewards before compute advantages"""
    gradient_checkpointing: bool = False
    """Enable gradient checkpointing"""

    # GRPO specific changes
    use_process_supervision : bool = False
    """Enable process Supervision"""
    use_iterative_reward_model_training : bool = True
    """Enable iterative reward model training"""
    sampling_group_size : int = 4
    """Group size sampling from the old policy model"""
    sampling_strategy : str = "top_p"
    """Sampling strategy for the group size sampling, only support top_p and top_k"""
    sampling_strategy_top_p : float = 0.95
    """Sampling strategy parameter for the group size sampling"""
    sampling_strategy_top_k : int = 0
    """Sampling strategy parameter for top k sampling"""

    # computed hyperparameters at runtime; we use `tyro.conf.Suppress` to hide them from the help text
    is_encoder_decoder: Optional[tyro.conf.Suppress[bool]] = None
    """TO BE FILLED In RUNTIME: Whether the model is an encoder-decoder model"""
    is_peft_model: Optional[tyro.conf.Suppress[bool]] = None
    """TO BE FILLED In RUNTIME: Whether the model is a PEFT model"""
    backward_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: Number of samples optimized in an `optimizer.step()` call"""
    global_backward_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: the effective `backward_batch_size` across all processes"""
    global_batch_size: tyro.conf.Suppress[int] = None
    """TO BE FILLED In RUNTIME: the effective `batch_size` across all processes"""

    if optimize_cuda_cache is not None:
        warnings.warn(
            "The `optimize_cuda_cache` argument will be deprecated soon, please use `optimize_device_cache` instead."
        )

        if optimize_device_cache is True:
            raise ValueError("Both `optimize_device_cache` and `optimize_cuda_cache` were provided")

        optimize_device_cache = optimize_cuda_cache

    def __post_init__(self):
        if self.forward_batch_size is not None:
            warnings.warn(
                "Note that using `forward_batch_size` is deprecated, use `mini_batch_size` instead. By setting it you overwrite `mini_batch_size` which affects both the batch size during forward passes and also the mini batch size for GRPO optimization."
            )
            self.mini_batch_size = self.forward_batch_size

        self.backward_batch_size = self.mini_batch_size * self.gradient_accumulation_steps
        exact_div(
            self.batch_size,
            self.backward_batch_size,
            "`batch_size` must be a multiple of `mini_batch_size * gradient_accumulation_steps`",
        )

        # check if wandb is installed
        if self.log_with == "wandb":
            # raise error if wandb is not installed
            if not is_wandb_available():
                raise ImportError(
                    "Please install wandb to use wandb logging. You can do this by running `pip install wandb`."
                )

        if self.sampling_strategy not in ["top_p", "top_k"]:
            raise ValueError("Sampling strategy must be either top_p or top_k")

        self.total_grpo_epochs = int(np.ceil(self.steps / self.batch_size))
        assert self.kl_penalty in ["kl", "abs", "mse", "full"]

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
