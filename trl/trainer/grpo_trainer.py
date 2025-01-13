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
)

from ..data_utils import maybe_apply_chat_template
from ..models import create_reference_model
from .grpo_config import GRPOConfig


class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
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
    ):
        self.ref_model = create_reference_model(model)
        self.reward_model = reward_model

        if data_collator is None:
            # No data collation is needed in GRPO
            def data_collator(features):
                return features

        self.max_completion_length = 256  # = |o_i| = L_c
        self.num_generations = 8  # = G
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.eos_token_id,
        )
        self.beta = 0.04

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

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

        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        examples = [maybe_apply_chat_template(example, self.processing_class) for example in inputs]
        prompts = [example["prompt"] for example in examples]
        inputs = self.processing_class(prompts, return_tensors="pt", padding=True)
        return super()._prepare_inputs(inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        assert return_outputs is False, "The GRPOTrainer does not support returning outputs"
        # Generate completions
        prompt_completion_ids = model.generate(**inputs, generation_config=self.generation_config)  # Shape (B*G, L)
        prompt_ids = prompt_completion_ids[:, : -self.max_completion_length]
        completion_ids = prompt_completion_ids[:, -self.max_completion_length :]

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids):
            logits = model(input_ids).logits
            logits = torch.roll(logits, shifts=1, dims=1)  # Shape (B*G, L)
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
            return per_token_logps

        prompt_len = prompt_ids.size(1)
        per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        per_token_logps = per_token_logps[:, prompt_len:]  # get rid of the prompt

        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_len:]  # get rid of the prompt

        # Compute the KL divergence between the model and the reference model
        per_token_kl = (
            torch.exp(ref_per_token_logps) / torch.exp(per_token_logps) - ref_per_token_logps + per_token_logps - 1
        )

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device="cuda")
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device="cuda").expand(is_eos.size(0), -1)
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

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        per_token_advantages = (per_token_reward - mean_grouped_rewards) / std_grouped_rewards
        per_token_advantages = per_token_advantages[:, prompt_len:]  # get rid of the prompt

        # Compute the loss
        per_token_loss = -(per_token_advantages - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1)).mean()
        return loss
