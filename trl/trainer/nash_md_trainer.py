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

from ..models.modeling_base import GeometricMixtureWrapper
from ..models.utils import unwrap_model_for_generation
from .nash_md_config import NashMDConfig
from .online_dpo_trainer import OnlineDPOTrainer
from .utils import empty_cache, get_reward, truncate_right


if is_apex_available():
    from apex import amp


class NashMDTrainer(OnlineDPOTrainer):
    r"""
    Initialize NashMDTrainer as a subclass of [`OnlineDPOConfig`].

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
        args (`NashMDConfig`):
            The NashMD config arguments to use for training.
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

    _tag_names = ["trl", "nash-md"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        args: Optional[NashMDConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        peft_config: Optional[Dict] = None,
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
            peft_config=peft_config,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self._mixture_coef = self.args.mixture_coef

        # Overwrite the stats dictionary to include NashMD specific statistics
        self.stats = {
            # Remove "non_score_reward", "rlhf_reward", "scores_margin"
            # Add "mixture_coef"
            "loss/kl": [],
            "objective/entropy": [],
            "loss/score": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "val/model_contain_eos_token": [],
            "val/ref_contain_eos_token": [],
            "beta": [],
            "mixture_coef": [],
        }

    @property
    def mixture_coef(self):
        if isinstance(self._mixture_coef, list):
            epoch = self.state.epoch
            return self._mixture_coef[epoch] if epoch < len(self._mixture_coef) else self._mixture_coef[-1]
        else:
            return self._mixture_coef

    def _generate_completions(self, model, prompts):
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            model_output = unwrapped_model.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                generation_config=self.generation_config,
            )

            ref_model = model if self.ref_model is None else self.ref_model
            with torch.no_grad(), unwrap_model_for_generation(ref_model, self.accelerator) as unwrapped_ref_model:
                mixture_model = GeometricMixtureWrapper(
                    model=unwrapped_model,
                    ref_model=unwrapped_ref_model,
                    generation_config=self.generation_config,
                    mixture_coef=self.mixture_coef,
                    device=self.accelerator.device,
                )

                mixture_output = mixture_model.generate(
                    input_ids=prompts["input_ids"],
                    attention_mask=prompts["attention_mask"],
                    generation_config=self.generation_config,
                )

        return model_output, mixture_output

    def _process_completions(self, model_output, mixture_output, prompts):
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
        mixture_completion_ids = mixture_output[:, context_length:]
        mixture_completion_ids, mixture_completion_mask = truncate_right(
            mixture_completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        mixture_data = {
            "input_ids": torch.cat((prompts["input_ids"], mixture_completion_ids), dim=1),
            "attention_mask": torch.cat((prompts["attention_mask"], mixture_completion_mask), dim=1),
        }

        return model_data, mixture_data

    def _compute_rewards(self, model_data, mixture_data, context_length):
        with torch.no_grad():
            _, model_scores, _ = get_reward(
                self.reward_model, model_data["input_ids"], self.tokenizer.pad_token_id, context_length
            )
            _, mixture_scores, _ = get_reward(
                self.reward_model, mixture_data["input_ids"], self.tokenizer.pad_token_id, context_length
            )

        # Apply EOS penalty if needed
        if self.args.missing_eos_penalty is not None:
            model_contain_eos = torch.any(model_data["input_ids"] == self.tokenizer.eos_token_id, dim=-1)
            mixture_contain_eos = torch.any(mixture_data["input_ids"] == self.tokenizer.eos_token_id, dim=-1)
            model_scores[~model_contain_eos] -= self.args.missing_eos_penalty
            mixture_scores[~mixture_contain_eos] -= self.args.missing_eos_penalty

        return model_scores, mixture_scores

    def _compute_logprobs(self, model, model_data, context_length):
        def compute_logprobs_for_data(m, data):
            output = m(data["input_ids"], attention_mask=data["attention_mask"])
            logits = output.logits[:, context_length - 1 : -1]
            logprobs = F.log_softmax(logits, dim=-1)
            token_logprobs = torch.gather(logprobs, 2, data["input_ids"][:, context_length:].unsqueeze(-1)).squeeze(-1)
            return token_logprobs

        # Compute logprobs for model completions under the model
        model_logprobs_model_data = compute_logprobs_for_data(model, model_data)

        # Compute logprobs of model completions under the reference model
        with torch.no_grad():
            if self.ref_model is None:
                with model.disable_adapter():
                    ref_logprobs_model_data = compute_logprobs_for_data(model, model_data)
            else:
                ref_logprobs_model_data = compute_logprobs_for_data(self.ref_model, model_data)

        # Mask padding tokens
        model_padding_mask = model_data["attention_mask"][:, context_length:] == 0
        model_logprobs_model_data = model_logprobs_model_data.masked_fill(model_padding_mask, 0.0)
        ref_logprobs_model_data = ref_logprobs_model_data.masked_fill(model_padding_mask, 0.0)

        return (model_logprobs_model_data, ref_logprobs_model_data)

    def _compute_losses(
        self,
        model_logprobs_model_data,
        ref_logprobs_model_data,
        model_data_scores,
        mixture_data_scores,
    ):
        # Compute log probs
        model_logprobs_model_data_sum = model_logprobs_model_data.sum(1)
        ref_logprobs_model_data_sum = ref_logprobs_model_data.sum(1)

        # probability of the model data vs the mixture data
        probability = F.sigmoid(model_data_scores - mixture_data_scores)

        # reinforce score where 0.5 is a control variate
        score = (probability - 0.5) * model_logprobs_model_data_sum

        # kl divergence
        kl_div = model_logprobs_model_data_sum - ref_logprobs_model_data_sum

        # final loss
        loss = self.beta * kl_div - score

        return loss.mean(), score, kl_div

    def _log_statistics(
        self,
        model_data,
        mixture_data,
        model_logprobs_model_data,
        ref_logprobs_model_data,
        model_scores,
        mixture_scores,
        score,
        kl_div,
        context_length,
    ):
        # Helper function to gather and compute mean
        def gather_mean(tensor):
            return self.accelerator.gather(tensor).mean().item()

        # Log score
        self.stats["loss/score"].append(gather_mean(score))
        # Log KL divergence
        self.stats["loss/kl"].append(gather_mean(kl_div))

        # Log logprobs
        model_logprobs_model_data_sum = model_logprobs_model_data.sum(1)
        ref_logprobs_model_data_sum = ref_logprobs_model_data.sum(1)

        self.stats["logps/chosen"].append(gather_mean(model_logprobs_model_data_sum))
        self.stats["logps/rejected"].append(gather_mean(ref_logprobs_model_data_sum))

        # Log rewards
        self.stats["rewards/chosen"].append(gather_mean(model_scores))
        self.stats["rewards/rejected"].append(gather_mean(mixture_scores))

        # Calculate entropy for model data
        entropy_model_data = -model_logprobs_model_data.sum(1)
        self.stats["objective/entropy"].append(gather_mean(entropy_model_data))

        # Calculate margins
        margin = model_scores - mixture_scores
        self.stats["rewards/margins"].append(gather_mean(margin))

        # Calculate accuracy
        accuracy = (margin > 0).float()
        self.stats["rewards/accuracies"].append(gather_mean(accuracy))

        # Log EOS token statistics
        model_eos = (model_data["input_ids"][:, context_length:] == self.tokenizer.eos_token_id).any(dim=1)
        mixture_eos = (mixture_data["input_ids"][:, context_length:] == self.tokenizer.eos_token_id).any(dim=1)
        self.stats["val/model_contain_eos_token"].append(gather_mean(model_eos.float()))
        self.stats["val/ref_contain_eos_token"].append(gather_mean(mixture_eos.float()))

        # Log beta and mixture coef
        self.stats["beta"].append(self.beta)
        self.stats["mixture_coef"].append(self.mixture_coef)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        # need the prompt_ only
        inputs = self._prepare_inputs(inputs)
        context_length = inputs["prompt_input_ids"].shape[1]
        prompts = {
            "input_ids": inputs["prompt_input_ids"],
            "attention_mask": inputs["prompt_attention_mask"],
        }
        del inputs

        # Sample completions from both the model and the reference model
        model_output, mixture_output = self._generate_completions(model, prompts)

        # Process model completions
        model_data, mixture_data = self._process_completions(model_output, mixture_output, prompts)

        # Compute rewards
        model_data_scores, mixture_data_scores = self._compute_rewards(model_data, mixture_data, context_length)

        # Compute logprobs
        model_logprobs_model_data, ref_logprobs_model_data = self._compute_logprobs(model, model_data, context_length)

        # Compute loss
        loss, score, kl_div = self._compute_losses(
            model_logprobs_model_data, ref_logprobs_model_data, model_data_scores, mixture_data_scores
        )

        # Log everything
        self._log_statistics(
            model_data,
            mixture_data,
            model_logprobs_model_data.detach(),
            ref_logprobs_model_data,
            model_data_scores,
            mixture_data_scores,
            score.detach(),
            kl_div.detach(),
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
