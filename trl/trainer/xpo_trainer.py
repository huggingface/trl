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

from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig
from transformers.training_args import OptimizerNames
from transformers.utils import is_apex_available

from ..models.utils import unwrap_model_for_generation
from .online_dpo_trainer import OnlineDPOTrainer
from .utils import (
    empty_cache,
    get_reward,
    truncate_right,
)


if is_apex_available():
    from apex import amp


class XPOTrainer(OnlineDPOTrainer):
    _tag_names = ["trl", "xpo"]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        ref_model = self.ref_model
        ref_model.eval()

        # Sample completions from both the model and the reference model
        inputs = self._prepare_inputs(inputs)
        generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            min_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            use_cache=False if self.args.gradient_checkpointing else True,
        )
        num_examples, context_length = inputs["prompt_input_ids"].shape
        prompt_ids = inputs["prompt_input_ids"]
        prompt_mask = inputs["prompt_attention_mask"]

        # Generate completions from the model
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            model_output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config,
            )

        # Generate completions from the reference model
        with torch.no_grad(), unwrap_model_for_generation(ref_model, self.accelerator) as unwrapped_ref_model:
            ref_output = unwrapped_ref_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config,
            )

        del inputs

        # Process model completions
        model_completion_ids = model_output[:, context_length:]
        model_completion_ids, model_completion_mask = truncate_right(
            model_completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        model_prompt_completion_ids = torch.cat((prompt_ids, model_completion_ids), dim=1)
        model_prompt_completion_mask = torch.cat((prompt_mask, model_completion_mask), dim=1)

        # Process reference model completions
        ref_completion_ids = ref_output[:, context_length:]
        ref_completion_ids, ref_completion_mask = truncate_right(
            ref_completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        ref_prompt_completion_ids = torch.cat((prompt_ids, ref_completion_ids), dim=1)
        ref_prompt_completion_mask = torch.cat((prompt_mask, ref_completion_mask), dim=1)

        # Get logprobs for model completions
        model_output = model(model_prompt_completion_ids, attention_mask=model_prompt_completion_mask)
        model_logits = model_output.logits[:, context_length - 1 : -1]
        model_logprobs = F.log_softmax(model_logits, dim=-1)
        model_completion_logprobs = torch.gather(model_logprobs, 2, model_completion_ids.unsqueeze(-1)).squeeze(-1)

        # Get logprobs for reference model completions
        with torch.no_grad():
            ref_output = ref_model(ref_prompt_completion_ids, attention_mask=ref_prompt_completion_mask)
        ref_logits = ref_output.logits[:, context_length - 1 : -1]
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        ref_completion_logprobs = torch.gather(ref_logprobs, 2, ref_completion_ids.unsqueeze(-1)).squeeze(-1)

        # Get logprobs for model on reference completions (for XPO loss)
        model_on_ref_output = model(ref_prompt_completion_ids, attention_mask=ref_prompt_completion_mask)
        model_on_ref_logits = model_on_ref_output.logits[:, context_length - 1 : -1]
        model_on_ref_logprobs = F.log_softmax(model_on_ref_logits, dim=-1)
        model_on_ref_completion_logprobs = torch.gather(
            model_on_ref_logprobs, 2, ref_completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Concatenate model and reference completions for reward computation
        all_prompt_completion_ids = torch.cat((model_prompt_completion_ids, ref_prompt_completion_ids), dim=0)
        # Get rewards
        with torch.no_grad():
            _, all_scores, _ = get_reward(
                self.reward_model, all_prompt_completion_ids, self.tokenizer.pad_token_id, context_length
            )
        # Split scores for model and reference completions
        model_scores, ref_scores = all_scores.split(num_examples)

        # Apply EOS penalty if needed
        if self.args.missing_eos_penalty is not None:
            model_contain_eos = torch.any(model_completion_ids == self.tokenizer.eos_token_id, dim=-1)
            ref_contain_eos = torch.any(ref_completion_ids == self.tokenizer.eos_token_id, dim=-1)
            model_scores[~model_contain_eos] -= self.args.missing_eos_penalty
            ref_scores[~ref_contain_eos] -= self.args.missing_eos_penalty

        # Mask padding tokens
        model_padding_mask = ~model_completion_mask.bool()
        ref_padding_mask = ~ref_completion_mask.bool()
        model_completion_logprobs = model_completion_logprobs.masked_fill(model_padding_mask, 0.0)
        ref_completion_logprobs = ref_completion_logprobs.masked_fill(ref_padding_mask, 0.0)
        model_on_ref_completion_logprobs = model_on_ref_completion_logprobs.masked_fill(ref_padding_mask, 0.0)

        # Compute log ratios
        model_logprobs_sum = model_completion_logprobs.sum(1)
        ref_logprobs_sum = ref_completion_logprobs.sum(1)
        log_ratios = model_logprobs_sum - ref_logprobs_sum

        # Compute losses
        chosen_mask = model_scores >= ref_scores
        dpo_losses = torch.where(
            chosen_mask, -F.logsigmoid(self.args.beta * log_ratios), -F.logsigmoid(-self.args.beta * log_ratios)
        )

        # XPO specific loss (using model logprobs on reference model generations)
        xpo_losses = self.args.alpha * model_on_ref_completion_logprobs.sum(1)

        # Total loss
        loss = (dpo_losses - xpo_losses).mean()  # Note the minus sign for xpo_losses

        # Log everything
        contain_eos_token = torch.any(model_completion_ids == self.tokenizer.eos_token_id, dim=-1)
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())

        chosen_indices = torch.where(chosen_mask)[0]
        rejected_indices = torch.where(~chosen_mask)[0]

        chosen_logprobs_sum = model_logprobs_sum[chosen_indices]
        rejected_logprobs_sum = model_logprobs_sum[rejected_indices]
        chosen_ref_logprobs_sum = ref_logprobs_sum[chosen_indices]
        rejected_ref_logprobs_sum = ref_logprobs_sum[rejected_indices]

        self.stats["logps/chosen"].append(self.accelerator.gather(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather(rejected_logprobs_sum).mean().item())
        self.stats["objective/scores"].append(self.accelerator.gather(all_scores.mean()).mean().item())

        kl = model_completion_logprobs - ref_completion_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather(mean_kl).mean().item())

        non_score_reward = (-self.args.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(self.accelerator.gather(mean_non_score_reward).mean().item())

        rlhf_reward = all_scores + non_score_reward.repeat(2)
        self.stats["objective/rlhf_reward"].append(self.accelerator.gather(rlhf_reward).mean().item())

        mean_entropy = -model_completion_logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather(mean_entropy).mean().item())

        scores_margin = torch.where(chosen_mask, model_scores - ref_scores, ref_scores - model_scores)
        self.stats["objective/scores_margin"].append(self.accelerator.gather(scores_margin.mean()).mean().item())

        chosen_rewards = self.args.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        rejected_rewards = self.args.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather(chosen_rewards.mean())
        gathered_rejected_rewards = self.accelerator.gather(rejected_rewards.mean())
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())

        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())

        # # XPO specific logging
        # self.stats["objective/xpo_loss"].append(self.accelerator.gather(xpo_losses).mean().item())

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
