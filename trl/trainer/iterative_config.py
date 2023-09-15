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
from dataclasses import dataclass, field
from typing import Optional

from ..core import flatten_dict


@dataclass
class IterativeConfig(object):
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
    step_batch_size: Optional[int] = field(
        default=32, metadata={"help": "Number of samples per optimisation step inside the step function"}
    )
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "Adam learning rate"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "The number of gradient accumulation steps"}
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

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)
