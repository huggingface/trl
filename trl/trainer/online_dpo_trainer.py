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

from datasets import Dataset, IterableDataset
from transformers import DataCollator, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import EvalPrediction

from ..experimental.online_dpo import OnlineDPOTrainer as _OnlineDPOTrainer
from .base_trainer import BaseTrainer  # noqa: F401 - for type hints
from .judges import BasePairwiseJudge  # noqa: F401 - for type hints
from .online_dpo_config import OnlineDPOConfig  # noqa: F401 - for type hints


class OnlineDPOTrainer(_OnlineDPOTrainer):
    r"""
    Initialize OnlineDPOTrainer.

    This class is now located in `trl.experimental.online_dpo`. Please update your imports to
    `from trl.experimental.online_dpo import OnlineDPOTrainer`. The current import path will be
    removed in TRL 0.29. For more information, see https://github.com/huggingface/trl/issues/4223.

    For full documentation, please refer to the [`trl.experimental.online_dpo.OnlineDPOTrainer`] class.
    """

    def __init__(
        self,
        model,
        ref_model=None,
        reward_funcs=None,
        judge: BasePairwiseJudge | None = None,
        args: OnlineDPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes=None,
        peft_config=None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ) -> None:
        warnings.warn(
            "The `OnlineDPOTrainer` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.online_dpo import OnlineDPOTrainer`. The current import path will be "
            "removed and no longer supported in TRL 0.29. For more information, see "
            "https://github.com/huggingface/trl/issues/4223.",
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
            reward_processing_classes=reward_processing_classes,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
