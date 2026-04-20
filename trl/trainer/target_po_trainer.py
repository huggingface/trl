# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import textwrap
from collections.abc import Callable

import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback
from transformers.utils import is_peft_available

from .grpo_trainer import EnvironmentFactory, GRPOTrainer, RewardFunc, RolloutFunc
from .target_po_config import TargetPOConfig
from .utils import get_config_model_id


if is_peft_available():
    from peft import PeftConfig, PeftModel
else:
    PeftConfig = None
    PeftModel = PreTrainedModel


class TargetPOTrainer(GRPOTrainer):
    """
    Trainer for Target Policy Optimization (TargetPO).

    TargetPO reuses the [`GRPOTrainer`] online rollout and reward stack, but fits the current policy to a frozen
    target distribution over sampled completions.

    See [`GRPOTrainer`] for the supported model, dataset, reward function, and rollout arguments.
    """

    _tag_names = ["trl", "tpo"]
    _name = "TargetPO"
    _paper = {
        "title": "Target Policy Optimization",
        "id": "2604.06159",
        "citation": textwrap.dedent("""\
            @misc{kaddour2026targetpolicyoptimization,
                title        = {{Target Policy Optimization}},
                author       = {Jean Kaddour},
                year         = 2026,
                eprint       = {arXiv:2604.06159},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc],
        args: TargetPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = TargetPOConfig(f"{model_name}-TargetPO")

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            tools=tools,
            rollout_func=rollout_func,
            environment_factory=environment_factory,
        )
