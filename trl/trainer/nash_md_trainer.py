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

import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from datasets import Dataset, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction

from ..experimental.nash_md import NashMDTrainer as ExperimentalNashMDTrainer
from .judges import BasePairwiseJudge
from .nash_md_config import NashMDConfig


class NashMDTrainer(ExperimentalNashMDTrainer):
    """
    Trainer for the Nash-MD method.

    <Deprecated version="0.25.0">

    This class has been moved to `trl.experimental.nash_md.NashMDTrainer` and will be removed in TRL 0.29.0.
    Please update your imports:
    ```python
    from trl.experimental.nash_md import NashMDTrainer
    ```

    For more details, see: https://github.com/huggingface/trl/issues/4223

    </Deprecated>
    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module = None,
        ref_model: PreTrainedModel | nn.Module = None,
        reward_funcs: PreTrainedModel | nn.Module | None = None,
        judge: BasePairwiseJudge | None = None,
        args: NashMDConfig | None = None,
        data_collator: Callable | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        peft_config: dict | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        warnings.warn(
            "NashMDTrainer has been moved to trl.experimental.nash_md.NashMDTrainer and will be removed from "
            "trl.trainer in TRL 0.29.0. Please update your imports to: "
            "`from trl.experimental.nash_md import NashMDTrainer`. "
            "For more details, see: https://github.com/huggingface/trl/issues/4223",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_funcs=reward_funcs,
            judge=judge,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
