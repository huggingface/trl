# CGPO Authors: Tengyu Xu, Eryk Helenowski, Karthik Abinav Sankararaman, Di Jin, Kaiyan Peng, Eric Han, Shaoliang Nie, Chen Zhu, Hejia Zhang, Wenxuan Zhou, Zhouhao Zeng, Yun He,Karishma Mandyam, Arya Talabzadeh, Madian Khabsa, Gabriel Cohen, Yuandong Tian, Hao Ma, Sinong Wang, Han Fang
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
import random
import textwrap
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from datasets import Dataset
from torch.utils.data import IterableDataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_apex_available,
    is_wandb_available,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available, logging

from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .cgpo_config import CGPOConfig
from .utils import (
    DataCollatorForChatML,
    batch_generation,
    disable_dropout_in_model,
    generate_model_card,
    get_reward,
    prepare_deepspeed,
    truncate_right,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model

if is_apex_available():
    from apex import amp

if is_wandb_available():
    import wandb


logger = logging.get_logger(__name__)


# should be removed when the mixture of judges PR is merged
class MixtureOfConstraintJudges:
    "Placeholder waiting for https://github.com/huggingface/trl/pull/2159 to be merged"

    def __init__(self, method: Literal["all_violated", "all_satisfied", "all_random"] = "all_random"):
        self.method = method

    def judge(self, prompts, completions=None, shuffle_order=None):
        if self.method == "all_violated":
            return [0 for _ in range(len(prompts))]
        elif self.method == "all_satisfied":
            return [1 for _ in range(len(prompts))]
        else:
            return [random.choice([0, 1]) for _ in range(len(prompts))]


class CGPOTrainer(Trainer):
    r"""
    Initialize the CGPOTrainer.

    Args:
        model (`transformers.PreTrainedModel` or `torch.nn.Module`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`transformers.PreTrainedModel` or `torch.nn.Module` or `None`):
            The reference model to use for training. If None is specified, the reference model will be created from
            the model.
        reward_model (`transformers.PreTrainedModel` or `torch.nn.Module` or `None`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        mixture_of_judges (`MixtureOfConstraintJudges`):
            The mixtures of judges to check if completions satisfy a set of contraints.
        args (`CGPOConfig`):
            The CGPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            Ignored and replaced by an instance of `DataCollatorForChatML`.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class (`PreTrainedprocessing_classBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        peft_config (`Dict`):
            The peft config to use for training.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "cgpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        ref_model: Union[PreTrainedModel, nn.Module, None] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        mixture_of_judges: Optional[MixtureOfConstraintJudges] = None,
        args: Optional[CGPOConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, either omit the `ref_model` argument or pass `None`."
            )

        self.ref_model = ref_model

        if reward_model is None:
            raise ValueError("`reward_model` must be provided.")
        else:
            self.reward_model = reward_model

        if mixture_of_judges is None:
            raise ValueError("`mixture_of_judges` must be provided.")
        else:
            self.moj = mixture_of_judges

        if args is None:
            raise ValueError("`args` must be provided.")

        # Check that the processing_class is provided
        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")

        # Convert to PEFT model if peft_config is provided
        if peft_config is not None:
            # Check if PEFT is available
            if not is_peft_available():
                raise ImportError(
                    "PEFT is not available and passed `peft_config`. Please install PEFT with "
                    "`pip install peft` to use it."
                )

            # If the model is already a PeftModel, we need to merge and unload it.
            # Further information here: https://huggingface.co/docs/trl/dpo_trainer#reference-model-considerations-with-peft
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            # Get peft model with the given config
            model = get_peft_model(model, peft_config)

        # Disable dropout in the model if specified
        if args.disable_dropout:
            disable_dropout_in_model(model)

        # Handle the ref_model
        # Usually, the user wants the ref model to be the initial version of the model. When using PEFT, it's easy to
        # get the ref model, as it's just the model with a disabled adapter. When not using PEFT, we need to create
        # the ref model from the model by copying it and disable the gradients and set it in evaluation mode.
        if self.ref_model is None:  # No ref model provided, the most common case
            if peft_config is None:
                self.ref_model = create_reference_model(model)  # copy, disable gradients, set eval mode
            else:
                self.ref_model = None  # we don't need a ref model here, we can just disable the adapter.
        else:  # rare case, the user provided a ref model
            self.ref_model = ref_model
            self.ref_model.eval()

        # Set the reward model in eval mode
        if self.reward_model is not None:
            self.reward_model.eval()

        if data_collator is not None:
            warnings.warn(
                "`CGPOTrainer` only supports training with a custom `DataCollatorForChatML. "
                "The data collator is replaced by a custom `DataCollatorForChatML`. "
                "The specified data collator will not be used."
            )

        data_collator = DataCollatorForChatML(processing_class, max_length=args.max_length)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=0,
            do_sample=True,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=processing_class.pad_token_id,
        )

        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if hasattr(model.generation_config, "eos_token_id") and model.generation_config.eos_token_id is not None:
            self.generation_config.eos_token_id = model.generation_config.eos_token_id

        self.k = args.k
        self.rlhf_optimizer = args.rlhf_optimizer
        self.beta = args.beta
        self.kl_threshold = args.kl_threshold
        self.lamb = args.lamb
        self.local_genscore_mini_batch_size = (
            args.local_genscore_mini_batch_size
            if args.local_genscore_mini_batch_size
            else args.per_device_train_batch_size
        )

        if (
            self.local_genscore_mini_batch_size < args.per_device_train_batch_size
            or self.local_genscore_mini_batch_size > (args.per_device_train_batch_size * self.k)
        ):
            raise ValueError(
                "`local_genscore_mini_batch_size` should be higher than `per_device_train_batch_size` and smaller than `per_device_train_batch_size*k`."
            )

        # to avoid divisions by 0
        self.epsilon = 1e-9
        self._tag_names.append(args.rlhf_optimizer)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # Placed after the super().__init__ because we need self.is_deepspeed_enabled and self.accelerator
        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.ref_model = prepare_deepspeed(self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16)
        else:
            self.reward_model = self.reward_model.to(self.accelerator.device)
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to(self.accelerator.device)

        self.stats = {
            "constraints/judgements": [],
            "constraints/rewards": [],
        }

    def _get_batch_logprobs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, context_length: int, use_ref_model: bool = False
    ) -> torch.Tensor:
        if not use_ref_model:
            logits = self.model(input_ids, attention_mask=attention_mask).logits

        else:
            if self.ref_model is not None:
                logits = self.ref_model(input_ids, attention_mask=attention_mask).logits
            else:
                with self.model.disable_adapter():
                    logits = self.model(input_ids, attention_mask=attention_mask).logits

        all_logprobs = F.log_softmax(logits, dim=-1)
        logprobs = torch.take_along_dim(all_logprobs, input_ids.unsqueeze(-1), dim=-1).squeeze(-1)
        logprobs = torch.masked_fill(logprobs, ~attention_mask.bool(), 0.0)

        return logprobs[:, context_length - 1 : -1].sum(1)

    def crpg_optimization(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        bs = inputs["bs"]
        mini_bs = self.local_genscore_mini_batch_size
        context_length = inputs["context_length"]
        prompt_completion_ids = inputs["prompt_completion_ids"]
        full_bs = prompt_completion_ids.shape[0]
        prompt_completion_mask = inputs["prompt_completion_mask"]
        judgements = inputs["judgements"]
        rewards = inputs["rewards"]
        completion_logprobs = inputs["completion_logprobs"]

        with torch.no_grad():
            _, baseline_rewards, _ = get_reward(
                self.reward_model,
                inputs["input_ids"],
                self.processing_class.pad_token_id,
                context_length,
            )
        baseline_rewards = baseline_rewards.repeat_interleave(repeats=self.k, dim=0)

        # compute the reference logprobs on generated samples
        ref_logprobss = []
        for i in range(0, full_bs, mini_bs):
            mini_batch_ids = prompt_completion_ids[i : i + mini_bs]
            mini_batch_mask = prompt_completion_mask[i : i + mini_bs]
            with torch.no_grad():
                mini_batch_ref_logprobs = self._get_batch_logprobs(
                    mini_batch_ids, mini_batch_mask, context_length, use_ref_model=True
                )
            ref_logprobss.append(mini_batch_ref_logprobs)

        ref_logprobs = torch.cat(ref_logprobss)

        # kl_div regularizer
        kl_div = completion_logprobs - ref_logprobs
        kl_div_regularizer = torch.clamp(1 - kl_div / self.kl_threshold, min=0)

        # compute the constrained calibrated regularize rewards
        regularized_rewards = judgements * torch.sigmoid(rewards - baseline_rewards) * kl_div_regularizer
        mean_regularized_rewards = regularized_rewards.mean()

        total_loss = torch.tensor(0.0, device=self.model.device)
        for i in range(0, full_bs, bs):
            # Compute loss on a batch of size `bs`, instead of the full batch (`bs` * self.k) to avoid OOM.
            mini_batch_prompt_completion_ids = prompt_completion_ids[i : i + bs]
            mini_batch_prompt_completion_mask = prompt_completion_mask[i : i + bs]
            mini_batch_regularized_rewards = regularized_rewards[i : i + bs]

            # compute kl_divergence
            logprobs = self._get_batch_logprobs(
                mini_batch_prompt_completion_ids, mini_batch_prompt_completion_mask, context_length
            )

            losses = -logprobs * (mini_batch_regularized_rewards - mean_regularized_rewards)
            loss = losses.mean() / self.k

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            total_loss += loss

        return total_loss

    def codpo_optimization(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        bs = inputs["bs"]
        context_length = inputs["context_length"]

        # reshaping per prompt
        judgements = inputs["judgements"].view(bs, self.k).bool()
        rewards = inputs["rewards"].view(bs, self.k)
        prompt_completion_ids = inputs["prompt_completion_ids"].view(bs, self.k, -1)
        prompt_completion_mask = inputs["prompt_completion_mask"].view(bs, self.k, -1)

        positive_masked_rewards = judgements * rewards
        negative_masked_rewards = ~judgements * rewards

        # get chosen and rejected completions
        chosen_idx = torch.argmax(positive_masked_rewards, dim=1)
        # handle cases where all generations satisfy constraints
        rejected_idx = torch.where(
            judgements.sum(dim=1) == self.k,
            torch.argmin(rewards, dim=1),
            torch.argmin(negative_masked_rewards, dim=1),
        )
        batch_indices = torch.arange(bs)
        chosen_prompt_completion_ids = prompt_completion_ids[batch_indices, chosen_idx]
        chosen_prompt_completion_mask = prompt_completion_mask[batch_indices, chosen_idx]
        rejected_prompt_completion_ids = prompt_completion_ids[batch_indices, rejected_idx]
        rejected_prompt_completion_mask = prompt_completion_mask[batch_indices, rejected_idx]

        # get the batch log probabilities from policy and ref
        chosen_logprobs = self._get_batch_logprobs(
            chosen_prompt_completion_ids, chosen_prompt_completion_mask, context_length
        )
        rejected_logprobs = self._get_batch_logprobs(
            rejected_prompt_completion_ids, rejected_prompt_completion_mask, context_length
        )
        with torch.no_grad():
            chosen_ref_logprobs = self._get_batch_logprobs(
                chosen_prompt_completion_ids, chosen_prompt_completion_mask, context_length, use_ref_model=True
            )
            rejected_ref_logprobs = self._get_batch_logprobs(
                rejected_prompt_completion_ids, rejected_prompt_completion_mask, context_length, use_ref_model=True
            )

        # computes the regularized dpo loss
        pi_logratios = chosen_logprobs - rejected_logprobs
        ref_logratios = chosen_ref_logprobs - rejected_ref_logprobs
        logits = pi_logratios - ref_logratios
        chosen_length = chosen_prompt_completion_mask[:, context_length:].sum(-1)
        # eqn (14) in the paper
        losses = -(F.logsigmoid(self.beta * logits) + self.lamb / chosen_length * chosen_logprobs)

        # Skip samples where no generations satisfy all constraints
        positive_completion_mask = judgements.sum(dim=1) != 0
        loss = (losses * positive_completion_mask).sum() / (positive_completion_mask.sum() + self.epsilon)

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss

    def crraft_optimization(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        prompt_baseline_ids = inputs["input_ids"]
        prompt_baseline_mask = inputs["attention_mask"]

        rewards = inputs["rewards"]
        judgements = inputs["judgements"]
        prompt_completion_ids = inputs["prompt_completion_ids"]
        prompt_completion_mask = inputs["prompt_completion_mask"]
        completion_logprobs = inputs["completion_logprobs"]

        bs = inputs["bs"]
        mini_bs = self.local_genscore_mini_batch_size
        context_length = inputs["context_length"]

        # get baseline rewards
        with torch.no_grad():
            _, baseline_rewards, _ = get_reward(
                self.reward_model,
                prompt_baseline_ids,
                self.processing_class.pad_token_id,
                inputs["context_length"],
            )

        if self.args.missing_eos_penalty is not None:
            baseline_ids = prompt_baseline_ids[:, context_length:]
            contain_eos_token = torch.any(baseline_ids == self.generation_config.eos_token_id, dim=-1)
            baseline_rewards[~contain_eos_token] -= self.args.missing_eos_penalty

        # compute kl div regularizer
        ref_logprobss = []
        for i in range(0, prompt_completion_ids.shape[0], mini_bs):
            mini_batch_ids = prompt_completion_ids[i : i + mini_bs]
            mini_batch_mask = prompt_completion_mask[i : i + mini_bs]
            with torch.no_grad():
                mini_batch_ref_logprobs = self._get_batch_logprobs(
                    mini_batch_ids, mini_batch_mask, context_length, use_ref_model=True
                )
            ref_logprobss.append(mini_batch_ref_logprobs)

        ref_logprobs = torch.cat(ref_logprobss)
        kl_div = completion_logprobs - ref_logprobs
        kl_div_regularizer = kl_div <= self.kl_threshold

        baseline_rewards = baseline_rewards.repeat_interleave(repeats=self.k, dim=0)
        # compute the contrainted regularized calibrated rewards
        regularized_rewards = (kl_div_regularizer * judgements * torch.sigmoid(rewards - baseline_rewards)).view(
            bs, self.k
        )

        # get the best completions and rewards per prompt
        best_completion_indices = torch.argmax(regularized_rewards, dim=-1)
        batch_indices = torch.arange(bs)
        best_prompt_completion_ids = prompt_completion_ids.view(bs, self.k, -1)[batch_indices, best_completion_indices]
        best_prompt_completion_mask = prompt_completion_mask.view(bs, self.k, -1)[
            batch_indices, best_completion_indices
        ]
        best_rewards = regularized_rewards[batch_indices, best_completion_indices]

        best_completion_logprobs = self._get_batch_logprobs(
            best_prompt_completion_ids, best_prompt_completion_mask, context_length
        )

        no_positive_completion_mask = best_rewards == 0
        if no_positive_completion_mask.any():
            # only compute the baseline logprobs if we need to (ie some prompts do not have any positive samples)
            # the baseline calibrated reward is always equal to 0.5 ie sigmoid(baseline_rewards - baselines_rewards)
            # Check that the baseline satisfy all constraints: judgements and kl
            baseline_judgements = torch.zeros_like(best_rewards, dtype=torch.bool)
            prompts_text = [text for i, text in enumerate(inputs["prompts_text"]) if no_positive_completion_mask[i]]
            baseline_completions_text = [
                text for i, text in enumerate(inputs["baseline_completions_text"]) if no_positive_completion_mask[i]
            ]
            no_positive_judgements = self.moj.judge(prompts_text, baseline_completions_text)

            baseline_judgements[no_positive_completion_mask] = torch.tensor(
                no_positive_judgements, device=self.model.device, dtype=torch.bool
            )

            baseline_logprobs = torch.zeros_like(best_completion_logprobs)
            baseline_logprobs[no_positive_completion_mask] = self._get_batch_logprobs(
                prompt_baseline_ids[no_positive_completion_mask],
                prompt_baseline_mask[no_positive_completion_mask],
                context_length,
            )
            with torch.no_grad():
                baseline_ref_logprobs = torch.zeros_like(best_completion_logprobs)
                baseline_ref_logprobs[no_positive_completion_mask] = self._get_batch_logprobs(
                    prompt_baseline_ids[no_positive_completion_mask],
                    prompt_baseline_mask[no_positive_completion_mask],
                    context_length,
                    use_ref_model=True,
                )

                baseline_kl_div = baseline_logprobs - baseline_ref_logprobs

            baseline_kl_div_regularizer = baseline_kl_div <= self.kl_threshold
            replace_by_baseline_mask = no_positive_completion_mask * baseline_kl_div_regularizer * baseline_judgements

            best_completion_logprobs = torch.where(
                replace_by_baseline_mask, baseline_logprobs, best_completion_logprobs
            )
            # calibrated reward for the baseline is always 0.5 ie sigmoid(0)
            best_rewards = torch.where(replace_by_baseline_mask, 0.5, best_rewards)

        # compute loss as done in eqn (18) of the CGPO paper: https://huggingface.co/papers/2409.20370
        losses = -best_completion_logprobs * best_rewards
        # simulate skipping samples instead of using .mean()
        loss = losses.sum() / ((rewards != 0).sum() + self.epsilon)

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        bs, context_length = inputs["prompts"].shape
        inputs["prompts_text"] = self.processing_class.batch_decode(inputs["prompts"])
        inputs["baseline_completions_text"] = self.processing_class.batch_decode(
            inputs["input_ids"][:, context_length:]
        )

        # step 4 of algorithm 1 of the CGPO paper: https://huggingface.co/papers/2409.20370
        prompt_ids = inputs["prompts"].repeat_interleave(repeats=self.k, dim=0)
        prompt_mask = inputs["prompt_attention_mask"].repeat_interleave(repeats=self.k, dim=0)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            query_responses_ids, completion_logits = batch_generation(
                unwrapped_model,
                prompt_ids,
                self.local_genscore_mini_batch_size,
                self.processing_class.pad_token_id,
                self.generation_config,
            )

        completion_ids = query_responses_ids[:, context_length:]
        completion_logprobs = F.log_softmax(completion_logits, dim=-1)
        completion_logprobs = (
            torch.take_along_dim(completion_logprobs, completion_ids.unsqueeze(-1), dim=-1).squeeze(-1).sum(-1)
        )

        query_responses = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # step 5 of algorithm 1 of the CGPO paper: https://huggingface.co/papers/2409.20370
        with torch.no_grad():
            prompt_repeated = [item for item in inputs["prompts_text"] for _ in range(self.k)]
            judgements = self.moj.judge(prompt_repeated, query_responses)

        completion_ids, completion_mask = truncate_right(
            completion_ids, self.generation_config.eos_token_id, self.processing_class.pad_token_id
        )

        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        rewards = []
        for i in range(0, prompt_completion_ids.shape[0], self.local_genscore_mini_batch_size):
            # Compute rewards on a mini-batch of size `bs`, instead of the full batch (`bs` * self.k)
            mini_batch_prompt_completion_ids = prompt_completion_ids[i : i + self.local_genscore_mini_batch_size]
            with torch.no_grad():
                _, mini_batch_rewards, _ = get_reward(
                    self.reward_model,
                    mini_batch_prompt_completion_ids,
                    self.processing_class.pad_token_id,
                    context_length,
                )

            rewards.append(mini_batch_rewards)

        rewards = torch.cat(rewards, dim=0)
        # Completions that do not contain an eos token id are penalized.
        if self.args.missing_eos_penalty is not None:
            contain_eos_token = torch.any(completion_ids == self.generation_config.eos_token_id, dim=-1)
            rewards[~contain_eos_token] -= self.args.missing_eos_penalty

        inputs["rewards"] = rewards
        inputs["judgements"] = torch.tensor(judgements, device=self.model.device, dtype=torch.float)
        inputs["bs"] = bs
        inputs["context_length"] = context_length
        inputs["prompt_completion_ids"] = prompt_completion_ids
        inputs["prompt_completion_mask"] = prompt_completion_mask
        inputs["completion_logprobs"] = completion_logprobs

        if self.rlhf_optimizer == "crraft":
            loss = self.crraft_optimization(inputs)
        elif self.rlhf_optimizer == "codpo":
            loss = self.codpo_optimization(inputs)
        elif self.rlhf_optimizer == "crpg":
            loss = self.crpg_optimization(inputs)
        else:
            raise ValueError(f"{self.rlhf_optimizer} not supported.", "Choose between `codpo`, `crraft` and `crpg`.")

        self.stats["constraints/judgements"].append(self.accelerator.gather(inputs["judgements"]).mean().item())
        self.stats["constraints/rewards"].append(self.accelerator.gather(inputs["rewards"]).mean().item())

        return loss.detach() / self.args.gradient_accumulation_steps

    # Same as Trainer.evaluate but log our metrics
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                logs[key] = sum(val) / len(val)
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `List[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        citation = textwrap.dedent(
            """\
        TO ADD
        """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="CGPO",
            trainer_citation=citation,
            paper_title="The Perfect Blend: Redefining RLHF with Mixture of Judges",
            paper_id="2409.20370",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))
