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
class PPOConfig:
    r"""
    Configuration class for the [`PPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[: -len(".py")]`):
            Name of this experiment.
        seed (`int`, *optional*, defaults to `0`):
            Random seed.
        log_with (`Optional[Literal["wandb", "tensorboard"]]`, *optional*, defaults to `None`):
            Log with either `"wandb"` or `"tensorboard"`. Check
            [tracking](https://huggingface.co/docs/accelerate/usage_guides/tracking) for more details.
        task_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of task to use - used only for tracking purposes.
        model_name (`Optional[str]`, *optional*, defaults to `"gpt2"`):
            Name of model to use - used only for tracking purposes.
        query_dataset (`Optional[str]`, *optional*, defaults to `"imdb"`):
            Name of dataset to query - used only for tracking purposes.
        reward_model (`Optional[str]`, *optional*, defaults to `"sentiment-analysis:lvwerra/distilbert-imdb"`):
            Reward model to use - used only for tracking purposes.
        remove_unused_columns (`bool`, *optional*, defaults to `True`):
            Remove unused columns from the dataset.
        tracker_kwargs (`JSONDict`, *optional*, defaults to `{}`):
            Keyword arguments for the tracker (e.g. `python ppo.py --tracker_kwargs='{"wandb": {"entity": "my_wandb_entity", "name": "my_exp_name"}}'`.
        accelerator_kwargs (`JSONDict`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator.
        project_kwargs (`JSONDict`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator project config (e.g. `logging_dir`).
        tracker_project_name (`str`, *optional*, defaults to `"trl"`):
            Name of project to use for tracking.
        push_to_hub_if_best_kwargs (`JSONDict`, *optional*, defaults to `{}`):
            Keyword arguments for pushing model to the hub during training (e.g. repo_id).
        steps (`int`, *optional*, defaults to `20000`):
            Number of training steps.
        learning_rate (`float`, *optional*, defaults to `1.41e-5`):
            Learning rate for the optimizer.
        adap_kl_ctrl (`bool`, *optional*, defaults to `True`):
            Use adaptive KL control, otherwise linear.
        init_kl_coef (`Optional[float]`, *optional*, defaults to `0.2`):
            Initial KL penalty coefficient (used for adaptive and linear control).
        kl_penalty (`Literal["kl", "abs", "mse", "full"]`, *optional*, defaults to `"kl"`):
            kl penalty options. Possible values are:

                - `"kl"`: model_logp - ref_logp
                - `"abs"`: abs(kl)
                - `"mse"`: mean squared error mse(kl)
                - `"full"`: the actual kl for all tokens in the distribution.

        target (`float`, *optional*, defaults to `6.0`):
            Target KL value for adaptive KL control.
        horizon (`float`, *optional*, defaults to `10000.0`):
            Horizon for adaptive KL control.
        gamma (`float`, *optional*, defaults to `1.0`):
            Gamma parameter for advantage calculation.
        lam (`float`, *optional*, defaults to `0.95`):
            Lambda parameter for advantage calculation.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Range for clipping in PPO policy gradient loss.
        cliprange_value (`float`, *optional*, defaults to `0.2`):
            Range for clipping values in loss calculation.
        vf_coef (`float`, *optional*, defaults to `0.1`):
            Scaling factor for value loss.
        batch_size (`int`, *optional*, defaults to `128`):
            Number of samples per optimisation step.
        forward_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            DEPRECATED: use `mini_batch_size` instead, which does the same thing.
        mini_batch_size (`int`, *optional*, defaults to `128`):
            Number of samples optimized in each mini batch.
        gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of gradient accumulation steps.
        world_size (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for distributed training.
        ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of optimisation epochs per batch of samples.
        optimize_device_cache (`bool`, *optional*, defaults to `False`):
            Optimize device cache for slightly more memory-efficient training.
        early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the PPO optimization loop early is the KL too high.
        target_kl (`float`, *optional*, defaults to `1.0`):
            Stop early if we exceed this value by over 50%.
        compare_steps (`int`, *optional*, defaults to `1`):
            Compare the current step with the previous `compare_steps` steps.
        ratio_threshold (`float`, *optional*, defaults to `10.0`):
            Skip mini-batches with high PPO ratios that can cause loss spikes.
        use_score_scaling (`bool`, *optional*, defaults to `False`):
            Use score scaling.
        use_score_norm (`bool`, *optional*, defaults to `False`):
            Use score normalization. Only applicable if `use_score_scaling` is True.
        score_clip (`Optional[float]`, *optional*, defaults to `None`):
            Score clipping.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whiten the rewards before computing advantages.
        is_encoder_decoder (`Optional[bool]`, *optional*, defaults to `None`):
            When using the `model_init` argument (callable) to instantiate the model instead of the `model` argument,
            you need to specify if the model returned by the callable is an encoder-decoder model.
        is_peft_model (`Optional[bool]`, *optional*, defaults to `None`):
            Whether the model is a PEFT model.
        backward_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Number of samples optimized in an `optimizer.step()` call.
        global_backward_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Effective `backward_batch_size` across all processes.
        global_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Effective `batch_size` across all processes.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
    """

    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    seed: int = 0
    log_with: Optional[Literal["wandb", "tensorboard"]] = None
    task_name: Optional[str] = None
    model_name: str = "gpt2"
    query_dataset: str = "imdb"
    reward_model: str = "sentiment-analysis:lvwerra/distilbert-imdb"
    remove_unused_columns: bool = True
    tracker_kwargs: JSONDict = field(default_factory=dict)
    accelerator_kwargs: JSONDict = field(default_factory=dict)
    project_kwargs: JSONDict = field(default_factory=dict)
    tracker_project_name: str = "trl"
    push_to_hub_if_best_kwargs: JSONDict = field(default_factory=dict)
    steps: int = 20000
    learning_rate: float = 1.41e-5
    adap_kl_ctrl: bool = True
    init_kl_coef: float = 0.2
    kl_penalty: Literal["kl", "abs", "mse", "full"] = "kl"
    target: float = 6.0
    horizon: float = 10000.0
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    batch_size: int = 128
    forward_batch_size: Optional[int] = None
    mini_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    world_size: tyro.conf.Suppress[int] = None
    ppo_epochs: int = 4
    max_grad_norm: Optional[float] = None
    optimize_cuda_cache: Optional[bool] = None
    optimize_device_cache: bool = False
    early_stopping: bool = False
    target_kl: float = 1.0
    compare_steps: int = 1
    ratio_threshold: float = 10.0
    use_score_scaling: bool = False
    use_score_norm: bool = False
    score_clip: Optional[float] = None
    whiten_rewards: bool = False
    gradient_checkpointing: bool = False
    is_encoder_decoder: Optional[tyro.conf.Suppress[bool]] = None
    is_peft_model: Optional[tyro.conf.Suppress[bool]] = None
    backward_batch_size: tyro.conf.Suppress[int] = None
    global_backward_batch_size: Optional[tyro.conf.Suppress[int]] = None
    global_batch_size: tyro.conf.Suppress[int] = None
    dataset_num_proc: Optional[int] = None

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
                "Note that using `forward_batch_size` is deprecated, use `mini_batch_size` instead. By setting it you overwrite `mini_batch_size` which affects both the batch size during forward passes and also the mini batch size for PPO optimization."
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

        self.total_ppo_epochs = int(np.ceil(self.steps / self.batch_size))
        assert self.kl_penalty in ["kl", "abs", "mse", "full"]

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
