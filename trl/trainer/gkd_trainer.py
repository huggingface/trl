# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, PreTrainedModel

from .gkd_config import GKDConfig
from .sft_trainer import SFTTrainer
from .utils import disable_dropout_in_model


class GKDTrainer(SFTTrainer):
    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[GKDConfig] = None,
        *sft_args,
        **kwargs,
    ):
        super().__init__(*sft_args, args=args, **kwargs)

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["torch_dtype"] = (
                teacher_model_init_kwargs["torch_dtype"]
                if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            )

        if isinstance(teacher_model, str):
            warnings.warn(
                "You passed a teacher model_id to the GKDTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        self.teacher_model = teacher_model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)
            if self.teacher_model is not None:
                disable_dropout_in_model(self.teacher_model)

        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(self.teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)

        self.loss_function = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)

        # compute forward KL divergence of student w.r.t teacher
        loss = self.loss_function(
            F.log_softmax(outputs_student.logits, dim=-1),
            F.softmax(outputs_teacher.logits, dim=-1),
        )

        # Return weighted student loss
        return (loss, outputs_student) if return_outputs else loss
