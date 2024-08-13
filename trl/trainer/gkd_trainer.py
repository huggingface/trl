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

import random
import warnings
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedModel

from ..models.utils import unwrap_model_for_generation
from .gkd_config import GKDConfig
from .sft_trainer import SFTTrainer
from .utils import disable_dropout_in_model


class GKDTrainer(SFTTrainer):
    _tag_names = ["trl", "gkd"]

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

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature

    @staticmethod
    def generalized_jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0, reduction="batchmean"):
        """
        Compute the Generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div.

        Args:
            student_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            beta: Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature: Softmax temperature (default: 1.0)
            reduction: Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        # Compute the interpolated log probabilities
        interpolated_log_probs = beta * teacher_log_probs + (1 - beta) * student_log_probs

        # Compute KL divergences using F.kl_div
        kl_teacher = F.kl_div(interpolated_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(interpolated_log_probs, student_log_probs, reduction="none", log_target=True)

        # Compute the Generalized Jensen-Shannon Divergence
        jsd = beta * kl_student + (1 - beta) * kl_teacher

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / (jsd.size(0) * jsd.size(1))
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"]
        )

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"]
            )

        # compute loss
        loss = 0
        batch_size = inputs["labels"].shape[0]
        for i in range(batch_size):
            index = (inputs["labels"] != -100)[i]
            student_logits = outputs_student.logits[i, index, :][-self.args.max_new_tokens_response :, :]
            teacher_logits = outputs_teacher.logits[i, index, :][-self.args.max_new_tokens_response :, :]
            loss += self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                beta=self.beta,
            )
        loss /= batch_size

        # Return loss
        return (loss, outputs_student) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        if random.random() >= self.lmbda:
            # On-policy: Generate outputs from the student model with temperature self.temperature
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generation_config = GenerationConfig(
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_k=0,
                )
                attention_mask = inputs["prompts"] != self.tokenizer.pad_token_id

                # generate output with respect to the prompt only
                generated_outputs = unwrapped_model.generate(
                    input_ids=torch.masked_fill(inputs["prompts"], ~attention_mask, self.tokenizer.pad_token_id),
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )

                # set input_ids to the generated output and attention_mask to 1 where the generated output is not the pad token
                inputs["input_ids"] = torch.where(
                    generated_outputs.sequences != -100, generated_outputs.sequences, self.tokenizer.pad_token_id
                )
                inputs["attention_mask"] = torch.where(generated_outputs.sequences != -100, 1, 0)
                # set all but the last self.args.max_new_tokens_response to -100  in labels
                labels = generated_outputs.sequences
                labels[:, : -self.args.max_new_tokens_response] = -100
                inputs["labels"] = labels

        inputs = self._prepare_inputs(inputs)
        model.train()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
