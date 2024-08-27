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
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedModel

from ..models import PreTrainedModelWrapper
from ..models.utils import unwrap_model_for_generation
from .gkd_config import GKDConfig
from .sft_trainer import SFTTrainer
from .utils import DataCollatorForChatML, disable_dropout_in_model


if is_deepspeed_available():
    import deepspeed


class GKDTrainer(SFTTrainer):
    _tag_names = ["trl", "gkd"]

    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[GKDConfig] = None,
        *sft_args,
        **kwargs,
    ):
        # add remove_unused_columns=False to the the dataclass args
        args.remove_unused_columns = False
        kwargs["data_collator"] = DataCollatorForChatML(tokenizer=kwargs["tokenizer"], max_length=args.max_seq_length)

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

        if args.disable_dropout:
            disable_dropout_in_model(self.model)
            if teacher_model is not None:
                disable_dropout_in_model(teacher_model)

        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature

    @staticmethod
    def generalized_jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0, reduction="batchmean"):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1) of https://arxiv.org/abs/2306.13649 for the definition.

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
        interpolated_log_probs = beta * student_log_probs + (1 - beta) * teacher_log_probs

        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        kl_teacher = F.kl_div(interpolated_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(interpolated_log_probs, student_log_probs, reduction="none", log_target=True)

        # Compute the Generalized Jensen-Shannon Divergence
        jsd = beta * kl_teacher + (1 - beta) * kl_student

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
            student_logits = outputs_student.logits[i, index, :][-self.args.max_new_tokens :, :]
            teacher_logits = outputs_teacher.logits[i, index, :][-self.args.max_new_tokens :, :]
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
        if random.random() <= self.lmbda:
            # On-policy: Generate outputs from the student model with temperature self.temperature
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generation_config = GenerationConfig(
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_k=0,
                )

                # generate output with respect to the prompt only
                generated_outputs = unwrapped_model.generate(
                    input_ids=inputs["prompts"],
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )

                input_ids = []
                labels = []
                for i, sequence in enumerate(generated_outputs.sequences):
                    eos_pos = (sequence == self.tokenizer.eos_token_id).nonzero()
                    if eos_pos.numel() > 0:
                        eos_pos = eos_pos[-1].item()
                        sequence[eos_pos + 1 :] = self.tokenizer.pad_token_id
                    else:
                        eos_pos = sequence.size(0)

                    # Set input_ids to the generated output
                    input_ids.append(sequence)

                    # Set the labels to the generated output with the prompt masked out (i.e., the teacher-forced part)
                    prompt_length = (inputs["prompts"][i] != self.tokenizer.pad_token_id).sum().item()
                    label = torch.full_like(sequence, -100)
                    label[prompt_length:eos_pos] = sequence[prompt_length:eos_pos]
                    labels.append(label)

                # Update inputs, labels, and attention_mask
                inputs["input_ids"] = torch.stack(input_ids)
                inputs["labels"] = torch.stack(labels)
                inputs["attention_mask"] = (inputs["input_ids"] != self.tokenizer.pad_token_id).long()

        inputs = self._prepare_inputs(inputs)
        model.train()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
