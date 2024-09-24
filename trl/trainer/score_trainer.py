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

from typing import Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .online_dpo_trainer import OnlineDPOTrainer
from .score_config import SCoREConfig


class SCoRETrainer(OnlineDPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_config = SCoREConfig() 
        
        # Add SCoRE-specific statistics
        self.stats.update({
            "loss/stage1": [],
            "kl_div/first_attempt": [],
            "reward/second_attempt": [],
        })

    def _generate_completions(self, model, prompts):
        with self.accelerator.unwrap_model(model) as unwrapped_model:
            # Generate first attempt
            first_attempt = unwrapped_model.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                generation_config=self.generation_config,
            )

            # Prepare input for second attempt
            second_attempt_prompt = self._prepare_second_attempt_prompt(prompts, first_attempt)
            
            # Generate second attempt
            second_attempt = unwrapped_model.generate(
                input_ids=second_attempt_prompt["input_ids"],
                attention_mask=second_attempt_prompt["attention_mask"],
                generation_config=self.generation_config,
            )

        return first_attempt, second_attempt

    def _prepare_second_attempt_prompt(self, prompts, first_attempt):
        context_length = prompts["input_ids"].shape[1]
        first_completion = first_attempt[:, context_length:]
        correction_instruction = self.tokenizer.encode(
            self.score_config.correction_instruction,
            return_tensors="pt",
            add_special_tokens=False
        ).repeat(prompts["input_ids"].shape[0], 1).to(first_attempt.device)

        second_attempt_input_ids = torch.cat([
            prompts["input_ids"],
            first_completion,
            correction_instruction
        ], dim=1)

        second_attempt_attention_mask = torch.ones_like(second_attempt_input_ids)

        return {
            "input_ids": second_attempt_input_ids,
            "attention_mask": second_attempt_attention_mask
        }

    def _compute_stage1_loss(self, model, ref_model, first_attempt, second_attempt, prompts):
        context_length = prompts["input_ids"].shape[1]

        # Compute logprobs for first attempt
        first_attempt_logits = model(first_attempt["input_ids"], attention_mask=first_attempt["attention_mask"]).logits
        first_attempt_logprobs = F.log_softmax(first_attempt_logits[:, context_length-1:-1], dim=-1)
        first_attempt_token_logprobs = torch.gather(
            first_attempt_logprobs, 2, first_attempt["input_ids"][:, context_length:].unsqueeze(-1)
        ).squeeze(-1)

        # Compute KL divergence for first attempt
        with torch.no_grad():
            ref_first_attempt_logits = ref_model(first_attempt["input_ids"], attention_mask=first_attempt["attention_mask"]).logits
            ref_first_attempt_logprobs = F.log_softmax(ref_first_attempt_logits[:, context_length-1:-1], dim=-1)

        kl_div = F.kl_div(first_attempt_logprobs, ref_first_attempt_logprobs.exp(), reduction='none').sum(-1)

        # Compute reward for second attempt
        second_attempt_reward = self._compute_rewards(second_attempt, context_length)

        # Compute loss
        kl_loss = self.score_config.kl_coef * kl_div.mean()
        reward_loss = -second_attempt_reward.mean()
        
        loss = kl_loss + reward_loss

        # Log statistics
        self.stats["loss/stage1"].append(loss.item())
        self.stats["kl_div/first_attempt"].append(kl_div.mean().item())
        self.stats["reward/second_attempt"].append(second_attempt_reward.mean().item())

        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        ref_model = self.ref_model
        ref_model.eval()

        inputs = self._prepare_inputs(inputs)
        prompts = {
            "input_ids": inputs["prompt_input_ids"],
            "attention_mask": inputs["prompt_attention_mask"],
        }

        # Generate completions (both first and second attempts)
        first_attempt, second_attempt = self._generate_completions(model, prompts)

        # Process completions
        first_attempt_data = self._process_completion(first_attempt, prompts)
        second_attempt_data = self._process_completion(second_attempt, prompts)

        # Compute Stage I loss
        loss = self._compute_stage1_loss(model, ref_model, first_attempt_data, second_attempt_data, prompts)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss.detach()

    def _process_completion(self, completion, prompts):
        context_length = prompts["input_ids"].shape[1]
        completion_ids = completion[:, context_length:]
        completion_ids, completion_mask = self.truncate_right(
            completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        return {
            "input_ids": torch.cat((prompts["input_ids"], completion_ids), dim=1),
            "attention_mask": torch.cat((prompts["attention_mask"], completion_mask), dim=1),
        }

    @staticmethod
    def truncate_right(tokens, eos_token_id, pad_token_id):
        eos_index = (tokens == eos_token_id).long().argmax(dim=-1)
        eos_index = torch.where(eos_index > 0, eos_index, tokens.shape[1])
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < eos_index[:, None]
        tokens = tokens.masked_fill(~mask, pad_token_id)
        return tokens, mask
