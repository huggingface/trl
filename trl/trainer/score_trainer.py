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
from transformers import PreTrainedTokenizerBase

from ..data_utils import maybe_apply_chat_template
from .online_dpo_trainer import OnlineDPOTrainer
from .utils import get_reward


class SCoRETrainer(OnlineDPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add SCoRE-specific statistics
        self.stats.update(
            {
                "kl_div/first_attempt": [],
                "reward/reward_diff": [],
                "loss/policy": [],
            }
        )

    def _generate_completions(self, model, prompts):
        unwrapped_model = self.accelerator.unwrap_model(model)
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
        correction_instruction = (
            self.tokenizer.encode(self.args.correction_instruction, return_tensors="pt", add_special_tokens=False)
            .repeat(prompts["input_ids"].shape[0], 1)
            .to(first_attempt.device)
        )

        second_attempt_input_ids = torch.cat([prompts["input_ids"], first_completion, correction_instruction], dim=1)

        second_attempt_attention_mask = torch.ones_like(second_attempt_input_ids)

        return {"input_ids": second_attempt_input_ids, "attention_mask": second_attempt_attention_mask}

    def _compute_stage1_loss(self, model, ref_model, first_attempt, second_attempt, prompts, ground_truth_completions):
        context_length = prompts["input_ids"].shape[1]

        # Compute logprobs for first attempt
        first_attempt_logits = model(first_attempt["input_ids"], attention_mask=first_attempt["attention_mask"]).logits
        first_attempt_logprobs = F.log_softmax(first_attempt_logits[:, context_length - 1 : -1], dim=-1)

        # Compute KL divergence for first attempt
        with torch.no_grad():
            ref_first_attempt_logits = ref_model(
                first_attempt["input_ids"], attention_mask=first_attempt["attention_mask"]
            ).logits
            ref_first_attempt_logprobs = F.log_softmax(ref_first_attempt_logits[:, context_length - 1 : -1], dim=-1)

        # Create a mask for non-padding tokens
        non_padding_mask = (first_attempt["input_ids"][:, context_length:] != self.tokenizer.pad_token_id).float()

        # Gather the log probabilities of the actual tokens
        first_attempt_tokens = first_attempt["input_ids"][:, context_length:]
        first_attempt_logprobs = torch.gather(first_attempt_logprobs, 2, first_attempt_tokens.unsqueeze(-1)).squeeze(-1)
        ref_first_attempt_logprobs = torch.gather(ref_first_attempt_logprobs, 2, first_attempt_tokens.unsqueeze(-1)).squeeze(-1)

        # Mask out padding tokens
        first_attempt_logprobs = torch.masked_fill(first_attempt_logprobs, ~non_padding_mask.bool(), 0)
        ref_first_attempt_logprobs = torch.masked_fill(ref_first_attempt_logprobs, ~non_padding_mask.bool(), 0)

        # Compute KL divergence
        kl_div = (first_attempt_logprobs - ref_first_attempt_logprobs) * non_padding_mask
        kl_div = kl_div.sum(-1).mean()

        # Compute reward for second attempt against ground truth
        reward_diff = self._compute_rewards(second_attempt, prompts, ground_truth_completions)

        # Compute REINFORCE loss with KL penalty
        policy_loss = -(first_attempt_logprobs.sum(-1) * reward_diff).mean()
        kl_loss = self.beta * kl_div

        loss = policy_loss + kl_loss

        # Log statistics
        self.stats["kl_div/first_attempt"].append(kl_div.mean().item())
        self.stats["reward/reward_diff"].append(reward_diff.mean().item())
        self.stats["loss/policy"].append(policy_loss.item())

        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        # Apply chat template and tokenize the input.
        # We do this on-the-fly to enable the use of reward models and policies with different tokenizers / chat templates.
        batch_size = len(next(iter(inputs.values())))
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        inputs = [maybe_apply_chat_template(x, self.tokenizer) for x in inputs]
        inputs = [self.tokenize_row(x, self.model.config.is_encoder_decoder, self.tokenizer) for x in inputs]
        inputs = self.data_collator(inputs)
        inputs = self._prepare_inputs(inputs)

        prompts = {
            "input_ids": inputs["prompt_input_ids"],
            "attention_mask": inputs["prompt_attention_mask"],
        }
        ground_truth_completions = {
            "input_ids": inputs["completion_input_ids"],
            "attention_mask": inputs["completion_attention_mask"],
        }

        # Generate completions (both first and second attempts)
        first_attempt, second_attempt = self._generate_completions(model, prompts)

        # Process completions
        first_attempt_data = self._process_completion(first_attempt, prompts)
        second_attempt_data = self._process_completion(second_attempt, prompts)

        # Compute Stage I loss
        loss = self._compute_stage1_loss(
            model, self.ref_model, first_attempt_data, second_attempt_data, prompts, ground_truth_completions
        )

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss.detach()

    def _compute_rewards(self, completions, prompts, ground_truth_completions):
        context_length = prompts["input_ids"].shape[1]
        with torch.no_grad():
            _, generated_scores, _ = get_reward(
                self.reward_model, completions["input_ids"], self.tokenizer.pad_token_id, context_length
            )

        # Compute scores for ground-truth completions
        ground_truth_input_ids = torch.cat([prompts["input_ids"], ground_truth_completions["input_ids"]], dim=1)
        _, ground_truth_scores, _ = get_reward(
            self.reward_model, ground_truth_input_ids, self.tokenizer.pad_token_id, context_length
        )

        if self.args.missing_eos_penalty is not None:
            completion_contain_eos = torch.any(completions["input_ids"] == self.tokenizer.eos_token_id, dim=-1)
            generated_scores[~completion_contain_eos] -= self.args.missing_eos_penalty
            ground_truth_contain_eos = torch.any(
                ground_truth_completions["input_ids"] == self.tokenizer.eos_token_id, dim=-1
            )
            ground_truth_scores[~ground_truth_contain_eos] -= self.args.missing_eos_penalty

        return generated_scores - ground_truth_scores

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

    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
        """Tokenize a single row from a SFT specific dataset."""
        if not is_encoder_decoder:
            prompt_tokens = tokenizer(feature["prompt"], add_special_tokens=False)
            # Add BOS token to head of prompt. Avoid adding if it's already there
            if tokenizer.bos_token_id is not None:
                prompt_len_input_ids = len(prompt_tokens["input_ids"])
                if prompt_len_input_ids == 0 or tokenizer.bos_token_id != prompt_tokens["input_ids"][0]:
                    prompt_tokens["input_ids"] = [tokenizer.bos_token_id] + prompt_tokens["input_ids"]
                    prompt_tokens["attention_mask"] = [1] + prompt_tokens["attention_mask"]

            # Tokenize the ground-truth completion
            completion_tokens = tokenizer(feature["completion"], add_special_tokens=False)
        else:
            prompt_tokens = tokenizer(feature["prompt"], add_special_tokens=True)
            completion_tokens = tokenizer(feature["completion"], add_special_tokens=False)

        return {
            "prompt_input_ids": prompt_tokens["input_ids"],
            "prompt_attention_mask": prompt_tokens["attention_mask"],
            "completion_input_ids": completion_tokens["input_ids"],
            "completion_attention_mask": completion_tokens["attention_mask"],
        }
