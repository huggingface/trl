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
from typing import Any, Callable, Optional, Type, Union

import torch
import torch.nn as nn
import torch.utils.data
from datasets import Dataset, IterableDataset
from transformers import (
    DataCollator,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from ..data_utils import maybe_apply_chat_template
from ..models import create_reference_model
from .grpo_config import GRPOConfig
from .utils import generate_model_card, get_comet_experiment_url


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb


class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: GRPOConfig = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[Type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[PeftConfig] = None,
    ):
        # Models
        # Trained model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Reward model
        self.reward_model = reward_model

        # Data loading and preprocessing
        if data_collator is None:

            def data_collator(features):  # No data collation is needed in GRPO
                return features

        # Training arguments
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.eos_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"kl": [], "reward": [], "advantage": []}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        examples = [maybe_apply_chat_template(example, self.processing_class) for example in inputs]
        prompts = [example["prompt"] for example in examples]
        inputs = self.processing_class(prompts, return_tensors="pt", padding=True)
        inputs = super()._prepare_inputs(inputs)

        # Generate completions
        prompt_completion_ids = model.generate(**inputs, generation_config=self.generation_config)  # Shape (B*G, L)
        prompt_length = inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids):
            logits = model(input_ids).logits
            logits = torch.roll(logits, shifts=1, dims=1)  # Shape (B*G, L)
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
            return per_token_logps

        per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        per_token_logps = per_token_logps[:, prompt_length:]  # get rid of the prompt

        with torch.no_grad():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
            else:
                with model.disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length:]  # get rid of the prompt

        # Compute the KL divergence between the model and the reference model
        per_token_kl = (
            torch.exp(ref_per_token_logps) / torch.exp(per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        )

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Compute the reward
        prompt_mask = inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)
        prompt_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        def get_per_token_reward(model, input_ids):
            base_model = getattr(model, model.base_model_prefix)  # usually base_model_prefix = "model"
            output = base_model(input_ids=input_ids, attention_mask=prompt_completion_mask, output_hidden_states=True)
            per_token_reward = model.score(output.hidden_states[-1]).squeeze(2)
            return per_token_reward

        per_token_reward = get_per_token_reward(self.reward_model, prompt_completion_ids)

        # Get the last True index in the mask
        flipped = torch.flip(prompt_completion_mask, dims=[1])
        final_idx = prompt_completion_mask.shape[1] - torch.argmax(flipped.int(), dim=1) - 1

        # Get the reward logits for the last token in the sequence
        final_rewards = per_token_reward[torch.arange(per_token_reward.size(0)), final_idx]

        # Compute grouped-wise rewards
        mean_grouped_rewards = final_rewards.view(-1, self.num_generations).mean(dim=1, keepdim=True)
        std_grouped_rewards = final_rewards.view(-1, self.num_generations).std(dim=1, keepdim=True)
        std_grouped_rewards = torch.where(  # avoid division by zero
            std_grouped_rewards < 1e-8, torch.tensor(1.0, device=device), std_grouped_rewards
        )

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        per_token_advantage = (per_token_reward - mean_grouped_rewards) / std_grouped_rewards
        per_token_advantage = per_token_advantage[:, prompt_length:]  # get rid of the prompt

        # Compute the loss
        per_token_loss = -(per_token_advantage - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(final_rewards).mean().item())

        mean_advantage = ((per_token_advantage * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["advantage"].append(self.accelerator.gather_for_metrics(mean_advantage).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val)/len(val) for key, val in self._metrics.items()} # average the metrics
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics = {key: [] for key in self._metrics}

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

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
