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
from typing import TYPE_CHECKING, Any, Literal

from ..experimental.kto import KTOTrainer as ExperimentalKTOTrainer


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class KTOTrainer(ExperimentalKTOTrainer):
    r"""
    Trainer for Kahneman-Tversky Optimization (KTO).

    <Deprecated version="0.25.0">

    This class has been moved to `trl.experimental.kto.KTOTrainer` and will be removed in TRL 0.29.0.
    Please update your imports:
    ```python
    from trl.experimental.kto import KTOTrainer
    ```

    For more details, see: https://github.com/huggingface/trl/issues/4223

    </Deprecated>
    """

    def __init__(
        self,
        model: "PreTrainedModel | nn.Module | str | None" = None,
        ref_model: "PreTrainedModel | nn.Module | str | None" = None,
        args: "KTOConfig | None" = None,
        train_dataset: "Dataset | None" = None,
        eval_dataset: "Dataset | dict[str, Dataset] | None" = None,
        processing_class: "PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None" = None,
        data_collator: "DataCollator | None" = None,
        model_init: "Callable[[], PreTrainedModel] | None" = None,
        callbacks: "list[TrainerCallback] | None" = None,
        optimizers: tuple["torch.optim.Optimizer", "torch.optim.lr_scheduler.LambdaLR"] = (None, None),
        preprocess_logits_for_metrics: "Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None" = None,
        peft_config: "dict | None" = None,
        compute_metrics: "Callable[[EvalLoopOutput], dict] | None" = None,
    ):
        warnings.warn(
            "KTOTrainer has been moved to trl.experimental.kto.KTOTrainer and will be removed from "
            "trl.trainer in TRL 0.29.0. Please update your imports to: "
            "`from trl.experimental.kto import KTOTrainer`. "
            "For more details, see: https://github.com/huggingface/trl/issues/4223",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
        )
