# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import textwrap
from typing import Any, Callable, Optional, Union

import jinja2
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import OptimizerNames
from transformers.utils import is_apex_available

from ..data_utils import is_conversational, maybe_apply_chat_template
from ..models.modeling_base import GeometricMixtureWrapper
from ..models.utils import unwrap_model_for_generation
from .judges import BasePairwiseJudge
from .nash_md_config import NashMDConfig
from .online_dpo_trainer import OnlineDPOTrainer
from .utils import (
    SIMPLE_CHAT_TEMPLATE,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    selective_log_softmax,
    truncate_right,
)


if is_apex_available():
    from apex import amp


if is_wandb_available():
    import wandb


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
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        peft_config (`dict`):
            The peft config to use for training.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "nash-md"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[NashMDConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            judge=judge,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_class=processing_class,  # for now, NashMDTrainer can't use any reward model
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
            "rewards/probabilities": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "val/model_contain_eos_token": [],
            "val/ref_contain_eos_token": [],
            "beta": [],
            "mixture_coef": [],
        }
        if self.reward_model is not None:
            self.stats["rewards/chosen"] = []
            self.stats["rewards/rejected"] = []

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
            model_completion_ids, self.processing_class.eos_token_id, self.processing_class.pad_token_id
        )
        model_data = {
            "input_ids": torch.cat((prompts["input_ids"], model_completion_ids), dim=1),
            "attention_mask": torch.cat((prompts["attention_mask"], model_completion_mask), dim=1),
            "raw": prompts["raw"],
        }

        # Process reference model completions
        mixture_completion_ids = mixture_output[:, context_length:]
        mixture_completion_ids, mixture_completion_mask = truncate_right(
            mixture_completion_ids, self.processing_class.eos_token_id, self.processing_class.pad_token_id
        )
        mixture_data = {
            "input_ids": torch.cat((prompts["input_ids"], mixture_completion_ids), dim=1),
            "attention_mask": torch.cat((prompts["attention_mask"], mixture_completion_mask), dim=1),
            "raw": prompts["raw"],
        }

        return model_data, mixture_data

    def _compute_rewards(self, model_data, mixture_data, context_length):
        with torch.no_grad():
            _, model_scores, _ = get_reward(
                self.reward_model, model_data["input_ids"], self.processing_class.pad_token_id, context_length
            )
            _, mixture_scores, _ = get_reward(
                self.reward_model, mixture_data["input_ids"], self.processing_class.pad_token_id, context_length
            )

        # Apply EOS penalty if needed
        if self.args.missing_eos_penalty is not None:
            model_contain_eos = torch.any(model_data["input_ids"] == self.processing_class.eos_token_id, dim=-1)
            mixture_contain_eos = torch.any(mixture_data["input_ids"] == self.processing_class.eos_token_id, dim=-1)
            model_scores[~model_contain_eos] -= self.args.missing_eos_penalty
            mixture_scores[~mixture_contain_eos] -= self.args.missing_eos_penalty

        return model_scores, mixture_scores

    def _compute_judge(self, model_data, mixture_data, context_length):
        prompts = model_data["raw"]
        model_data_completions = self.processing_class.batch_decode(
            model_data["input_ids"][:, context_length:], skip_special_tokens=True
        )
        model_data_completions = [completion.strip() for completion in model_data_completions]

        mixture_data_completions = self.processing_class.batch_decode(
            mixture_data["input_ids"][:, context_length:], skip_special_tokens=True
        )
        mixture_data_completions = [completion.strip() for completion in mixture_data_completions]
        if is_conversational({"prompt": prompts[0]}):
            model_data_completions = [
                [{"role": "assistant", "content": completion}] for completion in model_data_completions
            ]
            environment = jinja2.Environment()
            template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
            prompts = [template.render(messages=message) for message in prompts]
            model_data_completions = [template.render(messages=completion) for completion in model_data_completions]

            mixture_data_completions = [
                [{"role": "assistant", "content": completion}] for completion in mixture_data_completions
            ]
            mixture_data_completions = [
                template.render(messages=completion) for completion in mixture_data_completions
            ]

        probability = self.judge.judge(
            prompts,
            list(zip(model_data_completions, mixture_data_completions)),
            return_scores=True,
        )
        return torch.tensor(probability, device=model_data["input_ids"].device)

    def _compute_logprobs(self, model, model_data, context_length):
        def compute_logprobs_for_data(m, data):
            output = m(data["input_ids"], attention_mask=data["attention_mask"])
            logits = output.logits[:, context_length - 1 : -1]
            token_logprobs = selective_log_softmax(logits, data["input_ids"][:, context_length:])
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
        probability,
    ):
        # reinforce score where 0.5 is a control variate
        score = (probability - 0.5) * model_logprobs_model_data.sum(1)

        # kl divergence via reinforce
        with torch.no_grad():
            log_ratio = model_logprobs_model_data - ref_logprobs_model_data
            kl_div_log = log_ratio.sum(1)
        kl_div_loss = (log_ratio * model_logprobs_model_data).sum(1)

        # final loss
        loss = self.beta * kl_div_loss - score

        return loss.mean(), score, kl_div_log

    def _log_statistics(
        self,
        model_data,
        mixture_data,
        model_logprobs_model_data,
        ref_logprobs_model_data,
        probability,
        score,
        kl_div,
        context_length,
        model_scores=None,
        mixture_scores=None,
    ):
        # Helper function to gather and compute mean
        def gather_mean(tensor):
            return self.accelerator.gather_for_metrics(tensor).mean().item()

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
        if self.reward_model is not None:
            self.stats["rewards/chosen"].append(gather_mean(model_scores))
            self.stats["rewards/rejected"].append(gather_mean(mixture_scores))

        # Log probabilities
        self.stats["rewards/probabilities"].append(gather_mean(probability))

        # Calculate entropy for model data
        entropy_model_data = -model_logprobs_model_data.sum(1)
        self.stats["objective/entropy"].append(gather_mean(entropy_model_data))

        # Calculate margins
        margin = model_logprobs_model_data_sum - ref_logprobs_model_data_sum
        self.stats["rewards/margins"].append(gather_mean(margin))

        # Calculate accuracy
        accuracy = (margin > 0).float()
        self.stats["rewards/accuracies"].append(gather_mean(accuracy))

        # Log EOS token statistics
        model_eos = (model_data["input_ids"][:, context_length:] == self.processing_class.eos_token_id).any(dim=1)
        mixture_eos = (mixture_data["input_ids"][:, context_length:] == self.processing_class.eos_token_id).any(dim=1)
        self.stats["val/model_contain_eos_token"].append(gather_mean(model_eos.float()))
        self.stats["val/ref_contain_eos_token"].append(gather_mean(mixture_eos.float()))

        # Log beta and mixture coef
        self.stats["beta"].append(self.beta)
        self.stats["mixture_coef"].append(self.mixture_coef)

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        model.train()

        # Apply chat template and tokenize the input
        batch_size = len(next(iter(inputs.values())))
        prompts = inputs["prompt"]
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [self.tokenize_row(x, self.model.config.is_encoder_decoder, self.processing_class) for x in inputs]
        inputs = self.data_collator(inputs)

        # need the prompt_ only
        inputs = self._prepare_inputs(inputs)
        context_length = inputs["prompt_input_ids"].shape[1]
        prompts = {
            "input_ids": inputs["prompt_input_ids"],
            "attention_mask": inputs["prompt_attention_mask"],
            "raw": prompts,
        }
        del inputs

        # Sample completions from both the model and the reference model
        model_output, mixture_output = self._generate_completions(model, prompts)

        # Process model completions
        model_data, mixture_data = self._process_completions(model_output, mixture_output, prompts)

        # Compute rewards
        if self.reward_model is not None:
            model_scores, mixture_scores = self._compute_rewards(model_data, mixture_data, context_length)
            # probability of the model data vs the mixture data
            probability = F.sigmoid(model_scores - mixture_scores)
        else:
            model_scores, mixture_scores = None, None
            probability = self._compute_judge(model_data, mixture_data, context_length)

        # Compute logprobs
        model_logprobs_model_data, ref_logprobs_model_data = self._compute_logprobs(model, model_data, context_length)

        # Compute loss
        loss, score, kl_div = self._compute_losses(model_logprobs_model_data, ref_logprobs_model_data, probability)

        # Log everything
        self._log_statistics(
            model_data,
            mixture_data,
            model_logprobs_model_data.detach(),
            ref_logprobs_model_data,
            probability,
            score.detach(),
            kl_div.detach(),
            context_length,
            model_scores,
            mixture_scores,
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

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @inproceedings{munos2024nash,
            title        = {{Nash Learning from Human Feedback}},
            author       = {R{\'{e}}mi Munos and Michal Valko and Daniele Calandriello and Mohammad Gheshlaghi Azar and Mark Rowland and Zhaohan Daniel Guo and Yunhao Tang and Matthieu Geist and Thomas Mesnard and C{\\^{o}}me Fiegel and Andrea Michi and Marco Selvi and Sertan Girgin and Nikola Momchev and Olivier Bachem and Daniel J. Mankowitz and Doina Precup and Bilal Piot},
            year         = 2024,
            booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024, Vienna, Austria, July 21-27, 2024},
            publisher    = {OpenReview.net},
            url          = {https://openreview.net/forum?id=Y5AmNYiyCQ}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="Nash-MD",
            trainer_citation=citation,
            paper_title="Nash Learning from Human Feedback",
            paper_id="2312.00886",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
