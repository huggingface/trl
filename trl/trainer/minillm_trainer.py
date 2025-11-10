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

import textwrap
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import is_liger_kernel_available, is_peft_available

from ..models import prepare_deepspeed
from .minillm_config import MiniLLMConfig
from .grpo_trainer import GRPOTrainer, RolloutFunc, RewardFunc
from .utils import disable_dropout_in_model, empty_cache


if is_peft_available():
    from peft import PeftConfig

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


def dummy_reward_func(completions: list, **kwargs):
    # placeholder reward function when no reward function is provided
    return [1.0 for _ in completions]


class MiniLLMTrainer(GRPOTrainer):
    """Trainer for MiniLLM: Knowledge Distillation of Language Models.

    For details on MiniLLM, see the paper: [MiniLLM](https://huggingface.co/papers/2306.08543).

    Difference from GKD:
    1. Based on GRPOTrainer, easy for integrating with RLVR.
    2. Long-Range Dependency, same as Tinker.
    3. Efficient KL implementation.

    Args:
        model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Model to be trained, or the string identifier of the model to be instantiated from a pretrained model.
        teacher_model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Teacher model for knowledge distillation, or the string identifier of the model to be instantiated from a
            pretrained model.
        args ([`GKDConfig`], *optional*):
            Training arguments.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Data collator to batch samples from the dataset. It defaults to a [`DataCollatorForChatML`] using the
            `processing_class`.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for training.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
           Class to process the data.
        compute_metrics (`Callable`, *optional*):
            Function to compute metrics at evaluation. Must take in an [`~transformers.EvalPrediction`] and return a
            dictionary string to float.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Function to preprocess the logits before computing the metrics. Must take in the `logits` and `labels` and
            return the logits to be used for metrics computation.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the `model` will be
            wrapped with the specified PEFT adapter.
        formatting_func (`Callable`, *optional*):
            Function to format the dataset. Must take in an example and return an example.
    """

    _tag_names = ["trl", "minillm"]
    _name = "MiniLLM"
    _paper = {
        "title": "MiniLLM: Knowledge Distillation of Large Language Models",
        "id": "2306.08543",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
                    @inproceedings{
                        gu2024minillm,
                        title={Mini{LLM}: Knowledge Distillation of Large Language Models},
                        author={Yuxian Gu and Li Dong and Furu Wei and Minlie Huang},
                        booktitle={The Twelfth International Conference on Learning Representations},
                        year={2024},
                        url={https://openreview.net/forum?id=5h0qf7IBZZ}
                    }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: MiniLLMConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
    ):
        if reward_funcs is None:
            reward_funcs = [dummy_reward_func]
        
        super().__init__(
            model,
            reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
        )

        if data_collator is not None:
            self.data_collator = data_collator

        # Liger fused GKD loss (JSD)
        # self.use_liger_gkd_loss = False
        # if args.use_liger_kernel:
        #     self.liger_jsd_loss = LigerFusedLinearJSDLoss(
        #         beta=args.beta,
        #         ignore_index=-100,
        #         temperature=args.temperature,
        #         compiled=False,
        #     )
        #     self.use_liger_gkd_loss = True

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.temperature = args.temperature
        self.kd_temperature = args.kd_temperature
        self.single_step_decomposition = args.single_step_decomposition
        self.gamma = args.gamma
        self.length_normalization = args.length_normalization

    def _single_step_decomposition_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "batchmean"
    ):
        """
        Compute the MiniLLM loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.08543 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """
        reg_loss = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        )  # (batch_size, sequence_length)

        # Masking
        if mask is not None:
            reg_loss = reg_loss[mask]

        # Apply reduction
        if reduction == "batchmean":
            return reg_loss.sum() / mask.sum() if mask is not None else reg_loss.sum() / reg_loss.size(0)
        elif reduction == "sum":
            return reg_loss.sum()
        elif reduction == "mean":
            return reg_loss.mean()
        else:
            return reg_loss

    def _compute_advantage(
        self, 
        student_log_probs_on_labels: torch.Tensor,
        teacher_log_probs_on_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the advantage for Reverse KL Divergence.

        Mostly following https://github.com/microsoft/LMOps/blob/main/minillm/minillm/losses.py, L37-L49
        rewards[t] = teacher_log_probs_on_labels[t] - student_log_probs_on_labels[t]
        if length_normalization:
            lengths[t] = \sum_{i=t}^{T} \gamma^{i-t}
            advantages[t] = \sum_{i=t}^{T} \gamma^{i-t} * (R_i) / lengths[t]
        else:
            advantages[t] = \sum_{i=t}^{T} \gamma^{i-t} * (R_i)


        Args:
            student_log_probs_on_labels: Log probabilities of the student model on the labels.
                Shape: (batch_size, sequence_length)
            teacher_log_probs_on_labels: Log probabilities of the teacher model on the labels.
                Shape: (batch_size, sequence_length)
            mask: Optional mask to apply to the log probabilities. Shape: (batch_size, sequence_length)
        Returns:
            advantage: Computed advantage. Shape: (batch_size, sequence_length)
        """
        response_length = student_log_probs_on_labels.size(1)
        if mask is None:
            mask = torch.ones_like(student_log_probs_on_labels)
        mask = mask.float()
        student_log_probs_on_labels = student_log_probs_on_labels * mask
        teacher_log_probs_on_labels = teacher_log_probs_on_labels * mask

        rewards = (
            teacher_log_probs_on_labels - student_log_probs_on_labels
        )  # (batch_size, sequence_length)

        if self.gamma > 0.0:
            gamma_pow = torch.pow(self.gamma, torch.arange(response_length, device=rewards.device))

            advantages = rewards * gamma_pow
            advantages = advantages.flip(1).cumsum(dim=1).flip(1)

            if self.length_normalization:
                mask = torch.where(mask < 0.5, 1e-4, mask)
                lengths = mask * gamma_pow
                lengths = lengths.flip(1).cumsum(dim=1).flip(1)
                advantages = advantages / lengths
        else:
            advantages = rewards

        return advantages

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # compute student output
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompt_ids"].shape[1]
        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = input_ids[:, prompt_lengths:]

        # Apply temperature scaling
        student_logits = student_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        student_log_probs_on_labels = torch.gather(
            student_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)
        teacher_log_probs_on_labels = torch.gather(
            teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        mask = (shifted_labels != -100)

        reverse_kl_advantage = self._compute_advantage(
            student_log_probs_on_labels=student_log_probs_on_labels,
            teacher_log_probs_on_labels=teacher_log_probs_on_labels,
            mask=mask,
        )

        inputs["advantages"] = inputs["advantages"].unsqueeze(1) + reverse_kl_advantage

        # compute GRPO loss on verifiable reward
        loss = self._compute_loss(model, inputs)

        # compute loss
        if self.single_step_decomposition:
            single_step_decomposition_loss = self._single_step_decomposition_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                mask=mask,
            )

            loss += single_step_decomposition_loss

        # empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss