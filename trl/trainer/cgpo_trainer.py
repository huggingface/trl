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
import warnings
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from datasets import Dataset
from torch.utils.data import IterableDataset
from transformers import (
    DataCollator,
    GenerationConfig,
    PreTrainedTokenizerBase,
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
    batch_generation,
    disable_dropout_in_model,
    generate_model_card,
    get_reward,
    prepare_deepspeed,
    truncate_right,
    DataCollatorForChatML,
    DataCollatorForChatMLTestingCGPO,
    first_true_indices,
    pad_to_length,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model

if is_apex_available():
    from apex import amp

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)

import random


class MixtureOfConstraintJudges:
    "Placeholder waiting for https://github.com/huggingface/trl/pull/2159 to be merged"

    def judge(self, prompts, completions=None, shuffle_order=None):
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
        args (`OnlineDPOConfig`):
            The online DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
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
        data_collator: Optional[DataCollatorForChatML] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
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

        # Check that the tokenizer is provided
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided.")

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
        if ref_model is None:  # No ref model provided, the most common case
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

        if not isinstance(data_collator, DataCollatorForChatML):
            # Improve the warning messaeg
            warnings.warn(
                "`CPGOTrainer` only supports training with a `DataCollatorForChatML` collator instance. "
                f"Received `{type(data_collator).__name__}` instead. We will replace it with a `DataCollatorForChatML` collator."
            )
            data_collator = DataCollatorForChatMLTestingCGPO(tokenizer, max_length=args.max_length)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=0,
            do_sample=True,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=tokenizer.pad_token_id,
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
        self.max_kl = args.max_kl
        self.lamb = args.lamb

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
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

    def _get_batch_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(input_ids, attention_mask=attention_mask).logits

        with torch.no_grad():
            if self.ref_model is not None:
                ref_logits = self.ref_model(input_ids, attention_mask=attention_mask).logits
            else:
                with self.model.disable_adapter():
                    ref_logits = self.model(input_ids, attention_mask=attention_mask).logits

        all_logprobs = F.log_softmax(logits, dim=-1)
        all_ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        logprobs = torch.take_along_dim(all_logprobs, input_ids.unsqueeze(-1), dim=-1).squeeze(-1)
        ref_logprobs = torch.take_along_dim(all_ref_logprobs, input_ids.unsqueeze(-1), dim=-1).squeeze(-1)

        logprobs = torch.masked_fill(logprobs, ~attention_mask.bool(), 0.0)
        ref_logprobs = torch.masked_fill(ref_logprobs, ~attention_mask.bool(), 0.0)

        return logprobs[:, context_length - 1 : -1].sum(1), ref_logprobs[:, context_length - 1 : -1].sum(1)

    def crpg_optimization(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        bs = inputs["bs"]
        context_length = inputs["context_length"]

        with torch.no_grad():
            _, baseline_rewards, _ = get_reward(
                self.reward_model,
                inputs["prompt_baseline_ids"],
                self.tokenizer.pad_token_id,
                context_length,
            )

        baseline_rewards = baseline_rewards.repeat_interleave(repeats=self.k, dim=0)

        rewards = torch.tensor(inputs["rewards"], device=self.model.device)
        judgements = torch.tensor(inputs["judgements"], device=self.model.device)

        calibrated_rewards = torch.sigmoid(rewards - baseline_rewards)

        # compute kl_divergence
        logprobs, ref_logprobs = self._get_batch_logprobs(
            inputs["prompt_completion_ids"], inputs["prompt_completion_mask"], context_length
        )

        with torch.no_grad():
            # kl_div is used as a regularization term here
            kl_div = logprobs - ref_logprobs
        kl_div_regularization = torch.clamp(1 - kl_div / self.max_kl, min=0)
        calibrated_regularized_rewards = judgements * calibrated_rewards * kl_div_regularization

        losses = -logprobs * (calibrated_regularized_rewards - calibrated_regularized_rewards.mean())

        return losses.sum() / bs

    def crraft_optimization(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """Implementation of the Calibrated Regularized Reward Ranking Finetuning (CRRAFT) policy opttimizer."""

        bs = inputs["bs"]
        context_length = inputs["context_length"]
        prompt_baseline_ids = inputs["prompt_baseline_ids"]
        prompt_baseline_mask = inputs["prompt_baseline_mask"]
        prompt_completion_ids = inputs["prompt_completion_ids"]
        prompt_completion_mask = inputs["prompt_completion_ids"]

        # get baseline rewards & judgements
        with torch.no_grad():
            _, baseline_rewards, _ = get_reward(
                self.reward_model,
                inputs["prompt_baseline_ids"],
                self.tokenizer.pad_token_id,
                inputs["context_length"],
            )

        baseline_judgements = self.moj.judge(inputs["prompt"], inputs["completion"])

        # reshaping for filtering
        rewards = inputs["rewards"].view(bs, self.k)
        judgements = torch.tensor(inputs["judgements"], device=self.model.device).view(bs, self.k)
        prompt_completion_ids = prompt_completion_ids.view(bs, self.k, -1)
        prompt_completion_mask = prompt_completion_mask.view(bs, self.k, -1)

        # get constrained calibrated reward without kl
        calibrated_rewards = torch.sigmoid(rewards - baseline_rewards.unsqueeze(-1))
        masked_calibrated_rewards = judgements * calibrated_rewards
        # the baseline calibrated reward is always equal to 0.5 ie sigmoid(baseline_rewards - baselines_rewards)
        baseline_calibrated_rewards = 0.5

        best_idx = torch.argmax(masked_calibrated_rewards, dim=1)
        no_positive_completion = masked_calibrated_rewards.sum(dim=1) == 0
        use_baseline_mask = (no_positive_completion & (baseline_judgements != 0)).unsqueeze(-1)

        if use_baseline_mask.sum() != 0:
            # we need to pad the samples as both baseline and completions will be used in the same batch
            max_length = max(prompt_completion_ids.shape[-1], prompt_baseline_ids.shape[-1])
            prompt_completion_ids = pad_to_length(
                prompt_completion_ids, max_length, pad_value=self.tokenizer.pad_token_id
            )
            prompt_completion_mask = pad_to_length(prompt_completion_mask, max_length, pad_value=0)
            prompt_baseline_ids = pad_to_length(prompt_baseline_ids, max_length, pad_value=self.tokenizer.pad_token_id)
            prompt_baseline_mask = pad_to_length(prompt_baseline_mask, max_length, pad_value=0)

        # get best rewards and completions
        best_idx_reshaped = (
            best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, prompt_completion_ids.size(-1))
        )  # bs, 1, seq_len
        best_prompt_completion_ids = prompt_completion_ids.gather(1, best_idx_reshaped).squeeze(1)
        best_prompt_completion_mask = prompt_completion_mask.gather(1, best_idx_reshaped).squeeze(1)
        best_masked_calibrated_rewards = masked_calibrated_rewards.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)

        if use_baseline_mask.sum() != 0:
            # if no generations satisfy all constraints and baseline satisfy constraints, use baseline
            best_prompt_completion_ids = torch.where(
                use_baseline_mask, prompt_baseline_ids, best_prompt_completion_ids
            )

            best_prompt_completion_mask = torch.where(
                use_baseline_mask, prompt_baseline_mask, best_prompt_completion_mask
            )

            best_masked_calibrated_rewards = torch.where(
                use_baseline_mask.squeeze(1), baseline_calibrated_rewards, best_masked_calibrated_rewards
            )

        # filter all generations whoes KL-divergence is larger than a pre-defined threshold
        logprobs, ref_logprobs = self._get_batch_logprobs(
            best_prompt_completion_ids, best_prompt_completion_mask, context_length
        )

        with torch.no_grad():
            masked_kl_div = (logprobs - ref_logprobs) < self.max_kl
        filtered_calibrated_rewards = masked_kl_div * best_masked_calibrated_rewards

        # compute loss as done in eqn (18) of the CGPO paper: https://huggingface.co/papers/2409.20370
        losses = -logprobs * filtered_calibrated_rewards

        # simulate skipping samples instead of using .mean()
        return (
            losses.sum() / filtered_calibrated_rewards.sum()
            if filtered_calibrated_rewards.sum() != 0
            else losses.sum()
        )

    def codpo_optimization(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        bs = inputs["bs"]
        context_length = inputs["context_length"]

        rewards = torch.tensor(inputs["rewards"], device=self.model.device).view(bs, self.k)
        judgements = torch.tensor(inputs["judgements"], device=self.model.device, dtype=torch.bool).view(bs, self.k)

        prompt_completion_ids = inputs["prompt_completion_ids"].view(bs, self.k, -1)
        prompt_completion_mask = inputs["prompt_completion_mask"].view(bs, self.k, -1)

        positive_masked_rewards = judgements * rewards
        negative_masked_rewards = 1 - (~judgements * rewards)

        # get chosen and rejected completions
        chosen_idx = torch.argmax(positive_masked_rewards, dim=1)
        rejected_idx = torch.argmax(negative_masked_rewards, dim=1)
        chosen_prompt_completion_ids = prompt_completion_ids[torch.arange(bs), chosen_idx]
        chosen_prompt_completion_mask = prompt_completion_mask[torch.arange(bs), chosen_idx]
        rejected_prompt_completion_ids = prompt_completion_ids[torch.arange(bs), rejected_idx]
        rejected_prompt_completion_mask = prompt_completion_mask[torch.arange(bs), rejected_idx]

        # get the batch log probabilities
        chosen_logprobs, chosen_ref_logprobs = self._get_batch_logprobs(
            chosen_prompt_completion_ids, chosen_prompt_completion_mask, context_length
        )
        rejected_logprobs, rejected_ref_logprobs = self._get_batch_logprobs(
            rejected_prompt_completion_ids, rejected_prompt_completion_mask, context_length
        )

        pi_logratios = chosen_logprobs - rejected_logprobs
        ref_logratios = chosen_ref_logprobs - rejected_ref_logprobs

        logits = pi_logratios - ref_logratios
        completion_only_mask = chosen_prompt_completion_mask[:, context_length:].bool()
        chosen_length = first_true_indices(~completion_only_mask)
        # eqn (14) in the paper
        losses = -(F.logsigmoid(self.beta * logits) + self.lamb / chosen_length * chosen_logprobs)

        return losses.mean()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        bs, context_length = inputs["prompt_ids"].shape

        # step 4 of algorithm 1 of the CGPO paper: https://huggingface.co/papers/2409.20370
        prompt_ids = inputs["prompt_ids"].repeat_interleave(repeats=self.k, dim=0)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            query_responses_ids, _ = batch_generation(
                unwrapped_model,
                prompt_ids,
                self.k,
                self.tokenizer.pad_token_id,
                self.generation_config,
            )

        completion_ids = query_responses_ids[:, context_length:]

        query_responses = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # step 5 of algorithm 1 of the CGPO paper: https://huggingface.co/papers/2409.20370
        with torch.no_grad():
            prompt_repeated = [item for item in inputs["prompt"] for _ in range(self.k)]
            judgements = self.moj.judge(prompt_repeated, query_responses)

        completion_ids, completion_mask = truncate_right(
            completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_ids != self.tokenizer.pad_token_id, completion_mask), dim=1)

        rewards = []
        for i in range(0, prompt_completion_ids.shape[0], self.k):
            mini_batch_prompt_completion_ids = prompt_completion_ids[i : i + self.k]
            with torch.no_grad():
                _, mini_batch_rewards, _ = get_reward(
                    self.reward_model,
                    mini_batch_prompt_completion_ids,
                    self.tokenizer.pad_token_id,
                    context_length,
                )

            rewards.append(mini_batch_rewards)

        # TO DO: Add penalty for samples that do not contain the eos token
        rewards = torch.cat(rewards, dim=0)

        inputs["rewards"] = rewards
        inputs["judgements"] = judgements
        inputs["bs"] = bs
        inputs["context_length"] = context_length
        inputs["prompt_completion_ids"] = prompt_completion_ids
        inputs["prompt_completion_mask"] = prompt_completion_mask

        if self.rlhf_optimizer == "crraft":
            loss = self.crraft_optimization(inputs)
        elif self.rlhf_optimizer == "codpo":
            loss = self.codpo_optimization(inputs)
        elif self.rlhf_optimizer == "crpg":
            loss = self.crpg_optimization(inputs)
        else:
            raise ValueError(f"{self.rlhf_optimizer} not supported.", "Choose between `codpo`, `crraft` and `crpg`.")

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

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
