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
import logging
import os
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests

from ..core import flatten_dict


def autotag() -> str:
    wandb_tag = ""
    logging.info("autotag feature is enabled")
    try:
        git_tag = subprocess.check_output(["git", "describe", "--tags"]).decode("ascii").strip()
        wandb_tag = f"{git_tag}"
        logging.info(f"identified git tag: {git_tag}")
    except subprocess.CalledProcessError:
        return wandb_tag

    git_commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"]).decode("ascii").strip()
    try:
        # if the current branch is not main, try find the PR number
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
        if git_branch != "main":
            # try finding the pull request number on github
            prs = requests.get(f"https://api.github.com/search/issues?q=repo:lvwerra/trl+is:pr+{git_commit}")
            if prs.status_code == 200:
                prs = prs.json()
                if len(prs["items"]) > 0:
                    pr = prs["items"][0]
                    pr_number = pr["number"]
                    wandb_tag += f",pr-{pr_number}"
            logging.info(f"identified github pull request: {pr_number}")
        else:
            logging.info("current branch is main, not searching for pull request")
    except Exception as e:
        logging.warning(f"Automatic autotag failed with the following error: {e}")

    return wandb_tag


@dataclass
class PPOConfig(object):
    """
    Configuration class for PPOTrainer
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of task to use - used only for tracking purposes"},
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of model to use - used only for tracking purposes"},
    )
    steps: Optional[int] = field(default=20000, metadata={"help": "Number of training steps"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "Adam learning rate"})
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    kl_penalty: Optional[str] = field(
        default="kl",
        metadata={
            "help": "kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl) and 'mse': mean squared error mse(kl)."
        },
    )
    target: Optional[float] = field(default=6, metadata={"help": "Target KL value for adaptive KL control"})
    horizon: Optional[float] = field(default=10000, metadata={"help": "Horizon for adaptive KL control"})
    gamma: Optional[float] = field(default=1, metadata={"help": "Gamma parameter for advantage calculation"})
    lam: Optional[float] = field(default=0.95, metadata={"help": "Lambda parameter for advantage calculation"})
    cliprange: Optional[float] = field(
        default=0.2, metadata={"help": "Range for clipping in PPO policy gradient loss"}
    )
    cliprange_value: Optional[float] = field(
        default=0.2, metadata={"help": "Range for clipping values in loss calculation"}
    )
    vf_coef: Optional[float] = field(default=0.1, metadata={"help": "Scaling factor for value loss"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "Number of samples per optimisation step"})
    forward_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples forward passed through model at a time"},
    )
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Number of samples optimized inside PPO together"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "The number of gradient accumulation steps"}
    )
    ppo_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "Number of optimisation epochs per batch of samples"},
    )
    remove_unused_columns: Optional[bool] = field(
        default=True,
        metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"},
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
    max_grad_norm: Optional[float] = field(
        default=None, metadata={"help": "Maximum gradient norm for gradient clipping"}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "Seed value for random generations"})
    optimize_cuda_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Optimize CUDA cache for slightly more memory-efficient training"},
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "Whether to stop the PPO optimization loop early is the KL too high"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "Stop early if we exceed this value by over 50%"}
    )
    push_to_hub_if_best_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for pushing model to the hub during training (e.g. repo_id)"},
    )
    compare_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of steps between comparison of the current reward with the best seen so far"},
    )
    ratio_threshold: Optional[float] = field(
        default=10.0, metadata={"help": "Skip mini-batches with high PPO ratios that can cause loss spikes"}
    )

    def __post_init__(self):
        if self.forward_batch_size is not None:
            warnings.warn(
                "Note that using `forward_batch_size` is deprecated, use `mini_batch_size` instead. By setting it you overwrite `mini_batch_size` which affects both the batch size during forward passes and also the mini batch size for PPO optimization."
            )
            self.mini_batch_size = self.forward_batch_size

        # check if wandb is installed
        if self.log_with == "wandb":
            # raise error if wandb is not installed
            try:
                import wandb  # noqa: F401

                existing_wandb_tag = os.environ.get("WANDB_TAGS", "")
                wandb_tag = autotag()
                if len(wandb_tag) > 0:
                    if len(existing_wandb_tag) > 0:
                        os.environ["WANDB_TAGS"] = ",".join([existing_wandb_tag, wandb_tag])
                    else:
                        os.environ["WANDB_TAGS"] = wandb_tag
                    logging.info(f"the following tags will be used for wandb logging: {os.environ['WANDB_TAGS']}")
            except ImportError:
                raise ImportError(
                    "Please install wandb to use wandb logging. You can do this by running `pip install wandb`."
                )

        self.total_ppo_epochs = int(np.ceil(self.steps / self.batch_size))
        assert self.kl_penalty in ["kl", "abs", "mse"]

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
