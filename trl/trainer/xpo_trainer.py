# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import OptimizerNames
from transformers.utils import is_apex_available

from ..models.utils import unwrap_model_for_generation
from .online_dpo_trainer import OnlineDPOTrainer
from .utils import (
    empty_cache,
    get_reward,
    truncate_right,
)
from .xpo_config import XPOConfig


if is_apex_available():
    from apex import amp


class XPOTrainer(OnlineDPOTrainer):
    r"""
    Initialize XPOTrainer as a subclass of [`OnlineDPOConfig`].

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        reward_model (`transformers.PreTrainedModel`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        judge (`BasePairwiseJudge`):
            The judge to use for pairwise comparison of model completions.
        args (`XPOConfig`):
            The XPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
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

    _tag_names = ["trl", "xpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        args: Optional[XPOConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Overwrite the stats dictionary to include XPO specific statistics
        self.stats = {
            # Remove "non_score_reward", "rlhf_reward", "scores"
            # Add "loss/dpo", "loss/xpo"
            "loss/dpo": [],
            "loss/xpo": [],
            "objective/kl": [],
            "objective/entropy": [],
            # Replace "scores" by "model_scores" and "ref_scores"
            "objective/model_scores": [],
            "objective/ref_scores": [],
            "objective/scores_margin": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            # Replace "contain_eos_token" by "model_contain_eos_token" and "ref_contain_eos_token"
            "val/model_contain_eos_token": [],
            "val/ref_contain_eos_token": [],
        }

    def _generate_completions(self, model, ref_model, prompts):
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            model_output = unwrapped_model.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                generation_config=self.generation_config,
            )

        with torch.no_grad(), unwrap_model_for_generation(ref_model, self.accelerator) as unwrapped_ref_model:
            ref_output = unwrapped_ref_model.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                generation_config=self.generation_config,
            )

        return model_output, ref_output

    def _process_completions(self, model_output, ref_output, prompts):
        context_length = prompts["input_ids"].shape[1]

        # Process model completions
        model_completion_ids = model_output[:, context_length:]
        model_completion_ids, model_completion_mask = truncate_right(
            model_completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        model_data = {
            "input_ids": torch.cat((prompts["input_ids"], model_completion_ids), dim=1),
            "attention_mask": torch.cat((prompts["attention_mask"], model_completion_mask), dim=1),
        }

        # Process reference model completions
        ref_completion_ids = ref_output[:, context_length:]
        ref_completion_ids, ref_completion_mask = truncate_right(
            ref_completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        ref_data = {
            "input_ids": torch.cat((prompts["input_ids"], ref_completion_ids), dim=1),
            "attention_mask": torch.cat((prompts["attention_mask"], ref_completion_mask), dim=1),
        }

        return model_data, ref_data

    def _compute_rewards(self, model_data, ref_data, context_length):
        all_input_ids = torch.cat([model_data["input_ids"], ref_data["input_ids"]], dim=0)

        with torch.no_grad():
            _, all_scores, _ = get_reward(
                self.reward_model, all_input_ids, self.tokenizer.pad_token_id, context_length
            )

        model_scores, ref_scores = all_scores.chunk(2)

        # Apply EOS penalty if needed
        if self.args.missing_eos_penalty is not None:
            model_contain_eos = torch.any(model_data["input_ids"] == self.tokenizer.eos_token_id, dim=-1)
            ref_contain_eos = torch.any(ref_data["input_ids"] == self.tokenizer.eos_token_id, dim=-1)
            model_scores[~model_contain_eos] -= self.args.missing_eos_penalty
            ref_scores[~ref_contain_eos] -= self.args.missing_eos_penalty

        return model_scores, ref_scores

    def _compute_logprobs(self, model, ref_model, model_data, ref_data, context_length):
        def compute_logprobs_for_data(m, data):
            output = m(data["input_ids"], attention_mask=data["attention_mask"])
            logits = output.logits[:, context_length - 1 : -1]
            logprobs = F.log_softmax(logits, dim=-1)
            token_logprobs = torch.gather(logprobs, 2, data["input_ids"][:, context_length:].unsqueeze(-1)).squeeze(-1)
            return token_logprobs

        # Compute logprobs for model completions
        model_logprobs_model_data = compute_logprobs_for_data(model, model_data)
        # Compute logprobs for model on reference completions (for XPO loss)
        model_logprobs_ref_data = compute_logprobs_for_data(model, ref_data)

        # Compute logprobs for reference model completions
        with torch.no_grad():
            ref_logprobs_model_data = compute_logprobs_for_data(ref_model, model_data)
            ref_logprobs_ref_data = compute_logprobs_for_data(ref_model, ref_data)

        # Mask padding tokens
        model_padding_mask = model_data["attention_mask"][:, context_length:] == 0
        ref_padding_mask = ref_data["attention_mask"][:, context_length:] == 0
        model_logprobs_model_data = model_logprobs_model_data.masked_fill(model_padding_mask, 0.0)
        model_logprobs_ref_data = model_logprobs_ref_data.masked_fill(ref_padding_mask, 0.0)
        ref_logprobs_ref_data = ref_logprobs_ref_data.masked_fill(ref_padding_mask, 0.0)
        ref_logprobs_model_data = ref_logprobs_model_data.masked_fill(model_padding_mask, 0.0)

        return model_logprobs_model_data, model_logprobs_ref_data, ref_logprobs_ref_data, ref_logprobs_model_data

    def _compute_losses(
        self,
        model_logprobs_model_data,
        model_logprobs_ref_data,
        ref_logprobs_ref_data,
        ref_logprobs_model_data,
        chosen_mask,
    ):
        # Compute log probs
        model_logprobs_model_data_sum = model_logprobs_model_data.sum(1)
        model_logprobs_ref_data_sum = model_logprobs_ref_data.sum(1)
        ref_logprobs_ref_data_sum = ref_logprobs_ref_data.sum(1)
        ref_logprobs_model_data_sum = ref_logprobs_model_data.sum(1)

        chosen_model_logprobs = torch.where(chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        chosen_ref_logprobs = torch.where(chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        chosen_log_ratios = chosen_model_logprobs - chosen_ref_logprobs

        rejected_model_logprobs = torch.where(~chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        rejected_ref_logprobs = torch.where(~chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        rejected_log_ratios = rejected_model_logprobs - rejected_ref_logprobs

        # Compute logits as the difference between chosen and rejected log ratios
        logits = chosen_log_ratios - rejected_log_ratios

        if self.args.loss_type == "sigmoid":
            dpo_losses = -F.logsigmoid(self.args.beta * logits)
        elif self.args.loss_type == "ipo":
            dpo_losses = (logits - 1 / (2 * self.args.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.args.loss_type}")

        # Compute XPO specific loss
        xpo_losses = self.args.alpha * model_logprobs_ref_data_sum

        # Total loss
        loss = (dpo_losses + xpo_losses).mean()

        return loss, dpo_losses, xpo_losses

    def _log_statistics(
        self,
        model_data,
        ref_data,
        model_logprobs_model_data,
        model_logprobs_ref_data,
        ref_logprobs_ref_data,
        ref_logprobs_model_data,
        model_scores,
        ref_scores,
        dpo_losses,
        xpo_losses,
        context_length,
    ):
        # Helper function to gather and compute mean
        def gather_mean(tensor):
            return self.accelerator.gather(tensor).mean().item()

        # Log losses
        self.stats["loss/dpo"].append(gather_mean(dpo_losses))
        self.stats["loss/xpo"].append(gather_mean(xpo_losses))

        # Log scores
        self.stats["objective/model_scores"].append(gather_mean(model_scores))
        self.stats["objective/ref_scores"].append(gather_mean(ref_scores))
        self.stats["objective/scores_margin"].append(gather_mean(model_scores - ref_scores))

        # Determine which model outputs are "chosen" vs "rejected"
        chosen_mask = model_scores >= ref_scores

        # Log logprobs
        model_logprobs_model_data_sum = model_logprobs_model_data.sum(1)
        model_logprobs_ref_data_sum = model_logprobs_ref_data.sum(1)
        ref_logprobs_ref_data_sum = ref_logprobs_ref_data.sum(1)
        ref_logprobs_model_data_sum = ref_logprobs_model_data.sum(1)

        chosen_model_logprobs = torch.where(chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        chosen_ref_logprobs = torch.where(chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        chosen_log_ratios = chosen_model_logprobs - chosen_ref_logprobs

        rejected_model_logprobs = torch.where(~chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        rejected_ref_logprobs = torch.where(~chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        rejected_log_ratios = rejected_model_logprobs - rejected_ref_logprobs

        self.stats["logps/chosen"].append(gather_mean(chosen_model_logprobs.mean() + chosen_ref_logprobs.mean()))
        self.stats["logps/rejected"].append(gather_mean(rejected_model_logprobs.mean() + rejected_ref_logprobs.mean()))

        # Log rewards
        # Compute various statistics
        chosen_rewards = chosen_log_ratios * self.args.beta
        rejected_rewards = rejected_log_ratios * self.args.beta
        self.stats["rewards/chosen"].append(gather_mean(chosen_rewards.mean()))
        self.stats["rewards/rejected"].append(gather_mean(rejected_rewards.mean()))

        # Calculate KL divergence for model and ref data
        kl_model_data = model_logprobs_model_data - ref_logprobs_model_data
        kl_ref_data = model_logprobs_ref_data - ref_logprobs_ref_data
        mean_kl = (kl_model_data.sum(1) + kl_ref_data.sum(1)).mean() / 2
        self.stats["objective/kl"].append(gather_mean(mean_kl))

        # Calculate entropy for model and ref data
        entropy_model_data = -model_logprobs_model_data.sum(1)
        entropy_ref_data = -model_logprobs_ref_data.sum(1)
        mean_entropy = (entropy_model_data.mean() + entropy_ref_data.mean()) / 2
        self.stats["objective/entropy"].append(gather_mean(mean_entropy))

        # Calculate margins
        margin = chosen_rewards - rejected_rewards
        self.stats["rewards/margins"].append(gather_mean(margin.mean()))

        # Calculate accuracy
        accuracy = (margin > 0).float()
        self.stats["rewards/accuracies"].append(gather_mean(accuracy.mean()))

        # Log EOS token statistics
        model_eos = (model_data["input_ids"][:, context_length:] == self.tokenizer.eos_token_id).any(dim=1)
        ref_eos = (ref_data["input_ids"][:, context_length:] == self.tokenizer.eos_token_id).any(dim=1)
        self.stats["val/model_contain_eos_token"].append(gather_mean(model_eos.float()))
        self.stats["val/ref_contain_eos_token"].append(gather_mean(ref_eos.float()))

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        ref_model = self.ref_model
        ref_model.eval()

        # need the prompt_ only
        inputs = self._prepare_inputs(inputs)
        context_length = inputs["prompt_input_ids"].shape[1]
        prompts = {
            "input_ids": inputs["prompt_input_ids"],
            "attention_mask": inputs["prompt_attention_mask"],
        }
        del inputs

        # Sample completions from both the model and the reference model
        model_output, ref_output = self._generate_completions(model, ref_model, prompts)

        # Process model completions
        model_data, ref_data = self._process_completions(model_output, ref_output, prompts)

        # Compute rewards
        model_data_scores, ref_data_scores = self._compute_rewards(model_data, ref_data, context_length)

        # Compute logprobs
        model_logprobs_model_data, model_logprobs_ref_data, ref_logprobs_ref_data, ref_logprobs_model_data = (
            self._compute_logprobs(model, ref_model, model_data, ref_data, context_length)
        )

        # Compute loss
        loss, dpo_losses, xpo_losses = self._compute_losses(
            model_logprobs_model_data,
            model_logprobs_ref_data,
            ref_logprobs_ref_data,
            ref_logprobs_model_data,
            model_data_scores >= ref_data_scores,
        )

        # Log everything
        self._log_statistics(
            model_data,
            ref_data,
            model_logprobs_model_data.detach(),
            model_logprobs_ref_data.detach(),
            ref_logprobs_ref_data,
            ref_logprobs_model_data,
            model_data_scores,
            ref_data_scores,
            dpo_losses.detach(),
            xpo_losses.detach(),
            context_length,
        )

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}
        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
