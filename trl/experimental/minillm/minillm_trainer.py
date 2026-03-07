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

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import is_peft_available

from ...models import prepare_deepspeed
from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc, RolloutFunc
from ...trainer.utils import disable_dropout_in_model, get_config_model_id
from ..utils import empty_cache
from .minillm_config import MiniLLMConfig
from ...extras.profiling import profiling_decorator


if is_peft_available():
    from peft import PeftConfig


def dummy_reward_func(completions: list, **kwargs):
    # placeholder reward function when no reward function is provided
    return [1.0 for _ in completions]


class MiniLLMTrainer(GRPOTrainer):
    """
    Trainer for the Knowledge Distillation of Language Models (MiniLLM) method. This algorithm was initially proposed
    in the paper [Knowledge Distillation of Large Language Models](https://huggingface.co/papers/2306.08543).

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.minillm import MiniLLMTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = MiniLLMTrainer(
        model="Qwen/Qwen3-0.6B",
        teacher_model="Qwen/Qwen3-1.7B",
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        teacher_model (`PreTrainedModel | nn.Module | str`):
            Teacher model used for knowledge distillation. Instantiated similarly to `model`.
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`experimental.minillm.MiniLLMConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It must take prompts, args, and processing_class as parameters
            and return a dict with `"prompt_ids"`, `"completion_ids"`, and `"logprobs"` fields. Any other fields that
            are forwarded to the reward functions. This feature is experimental and may change or be removed at any
            time without prior notice.
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
                title={{MiniLLM: Knowledge Distillation of Large Language Models}},
                author={Yuxian Gu and Li Dong and Furu Wei and Minlie Huang},
                booktitle={The Twelfth International Conference on Learning Representations},
                year={2024},
                url={https://openreview.net/forum?id=5h0qf7IBZZ}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        teacher_model: PreTrainedModel | nn.Module | str,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: MiniLLMConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
    ):
        if reward_funcs is None:
            reward_funcs = [dummy_reward_func]

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = MiniLLMConfig(f"{model_name}-MiniLLM")

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

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

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the MiniLLMConfig, but your teacher_model is already instantiated."
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
        self.rkl_advantage = args.rkl_advantage
        self.gamma = args.gamma
        self.length_normalization = args.length_normalization

        self._current_rkl_advantange_time = 0.0
        self._current_single_step_loss_time = 0.0
        self._current_rl_loss_time = 0.0

    def _single_step_decomposition_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "batchmean",
    ):
        """
        Compute the MiniLLM loss for knowledge distillation using F.kl_div. See Eq. (1) of
        https://huggingface.co/papers/2306.08543 for the definition.

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

    @torch.no_grad()
    def _compute_advantage(
        self,
        student_log_probs_on_labels: torch.Tensor,
        teacher_log_probs_on_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Compute the advantage for Reverse KL Divergence.

        Mostly following [this
        implementation](https://github.com/microsoft/LMOps/blob/e210d2c026b9958617887762400778ace81172e6/minillm/minillm/losses.py#L37-L49).

        $$ \text{rewards}_t = \text{teacher\_log\_probs\_on\_labels}_t - \text{student\_log\_probs\_on\_labels}_t $$

        If length normalization is enabled:

        $$ \text{lengths}_t = \sum_{i=t}^{T} \gamma^{i-t} $$

        $$ \text{advantages}_t = \frac{\sum_{i=t}^{T} \gamma^{i-t} \text{rewards}_i}{\text{lengths}_t} $$

        Otherwise:

        $$ \text{advantages}_t = \sum_{i=t}^{T} \gamma^{i-t} \text{rewards}_i $$

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

        rewards = teacher_log_probs_on_labels - student_log_probs_on_labels  # (batch_size, sequence_length)

        if self.gamma > 0.0:
            gamma_pow = torch.pow(self.gamma, torch.arange(response_length, device=rewards.device))

            rewards = rewards * gamma_pow
            rewards = rewards.flip(1).cumsum(dim=1).flip(1)

            if self.length_normalization:
                mask = torch.where(mask < 0.5, 1e-4, mask)
                lengths = mask * gamma_pow
                lengths = lengths.flip(1).cumsum(dim=1).flip(1)
                rewards = rewards / lengths

        advantages = rewards

        return advantages

    def get_rev_kl(self, log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor):
        log_ratio = (log_p - log_q) * mask
        kl = log_ratio.float().exp() - 1 - log_ratio
        return kl

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)

        # Compute student output
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Slice the logits for the generated tokens using the inputs["prompts"] lengths
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

        generate_every = self.args.steps_per_generation * self.num_iterations
        if self.args.on_policy_logq or self.args.gradient_accumulation_steps % generate_every == 0:
            student_log_probs_on_labels = torch.gather(
                student_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
            ).squeeze(-1).detach()
        else:
            student_log_probs_on_labels = inputs["old_per_token_logps"]

        teacher_log_probs_on_labels = torch.gather(
            teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        mask = attention_mask[:, prompt_lengths:].bool()

        # Logging Meitrcs
        mode = "train" if self.model.training else "eval"

        # Compute advantage for Reverse KL
        if self.rkl_advantage:
            rkl_advantage_time = time.perf_counter()
            reverse_kl_advantage = self._compute_advantage(
                student_log_probs_on_labels=student_log_probs_on_labels,
                teacher_log_probs_on_labels=teacher_log_probs_on_labels,
                mask=mask,
            )

            inputs["advantages"] = inputs["advantages"].unsqueeze(1) + reverse_kl_advantage
            mean_advantage = (inputs["advantages"] * mask).sum() / mask.sum()
            self._metrics[mode]["minillm/mean_advantage"].append(self.accelerator.gather(mean_advantage).mean().item())
            rkl_advantage_time_after = time.perf_counter()
            self._current_rkl_advantange_time += rkl_advantage_time_after - rkl_advantage_time
            if (self._step + 1) % self.current_gradient_accumulation_steps == 0:
                self._metrics[mode]["rkl_advantage_time"].append(self._current_rkl_advantange_time)
                self._current_rkl_advantange_time = 0.0

        # Compute GRPO loss on verifiable reward
        rl_loss_time = time.perf_counter()
        loss = self._compute_loss(model, inputs)
        self._metrics[mode]["rl_loss"].append(self.accelerator.gather(loss).mean().item())
        rl_loss_time_after = time.perf_counter()
        self._current_rl_loss_time += rl_loss_time_after - rl_loss_time
        if (self._step + 1) % self.current_gradient_accumulation_steps == 0:
            self._metrics[mode]["rl_loss_time"].append(self._current_rl_loss_time)
            self._current_rl_loss_time = 0.0

        # Compute loss
        if self.single_step_decomposition:
            single_step_loss_time = time.perf_counter()
            single_step_decomposition_loss = self._single_step_decomposition_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                mask=mask,
            )

            self._metrics[mode]["minillm/single_step_decomposition_loss"].append(self.accelerator.gather(single_step_decomposition_loss).mean().item())
            loss += (single_step_decomposition_loss / self.current_gradient_accumulation_steps)
            single_step_loss_time_after = time.perf_counter()
            self._current_single_step_loss_time += single_step_loss_time_after - single_step_loss_time
            if (self._step + 1) % self.current_gradient_accumulation_steps == 0:
                self._metrics[mode]["single_step_loss_time"].append(self._current_single_step_loss_time)
                self._current_single_step_loss_time = 0.0

        # Compute Reverse KL for logging
        with torch.no_grad():
            reverse_kl = self.get_rev_kl(
                log_p=teacher_log_probs_on_labels,
                log_q=student_log_probs_on_labels,
                mask=mask,
            )
            reverse_kl = reverse_kl.sum() / mask.sum()
            self._metrics[mode]["minillm/reverse_kl"].append(self.accelerator.gather(reverse_kl).mean().item())

        # Empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss
