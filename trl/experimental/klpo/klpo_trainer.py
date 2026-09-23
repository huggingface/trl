# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

from typing import Any

import torch
import torch.nn.functional as F

from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id, selective_log_softmax
from .klpo_config import KLPOConfig


class KLPOTrainer(GRPOTrainer):
    """
    Trainer for KL-Regularized Policy Optimization (KLPO).

    KLPO (https://yifanzhang-pro.github.io/KLPO/) is a critic-free, single-rollout method for off-policy reinforcement
    learning. This trainer implements the report's default route: **token regression + Monte Carlo KL (MC-KL)**. One
    complete response per prompt is sufficient; there is no same-prompt response group, learned value/normalizer model,
    importance-ratio multiplier, reward centering, or ratio clipping.

    Let `p` be the current policy, `q` the sampler (behavior) policy, and `R` the terminal reward. For each generated
    token `a_u`, and `M` auxiliary token draws `v_j ~ q` at the same prefix (sampled IID with replacement,
    independently of the rollout), the loss is:

    ```
    ell_u = log p(a_u) - log q(a_u)
    z_u   = log p(a_u) - mean_j log p(v_j)
    loss  = -mean_responses sum_tokens stopgrad(R - klpo_beta * ell_u) * z_u
    ```

    Tokens are summed without length normalization and averaged over complete responses. The sampler `q` is the
    training policy snapshot at generation time: right after generation the trainer scores the completions with the
    current model, draws the `M` auxiliary tokens per prefix from that distribution, and keeps those records (and the
    action log-probabilities) fixed while the batch is reused. Following the report, no vLLM importance-sampling
    multiplier is applied to the loss, and the group-relative `advantages` computed by [`GRPOTrainer`] are ignored in
    favor of the raw terminal reward.

    The changes w.r.t. [`GRPOTrainer`] are `_compute_loss`, plus a `_generate_and_score_completions` override that
    records the raw rewards and the MC auxiliary draws. Everything else (generation, reward computation, weight
    syncing, metric logging) is inherited unchanged.
    """

    _tag_names = ["trl", "klpo"]

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = KLPOConfig(f"{model_name.split('/')[-1]}-KLPO")

        super().__init__(model, reward_funcs, args=args, **kwargs)

        self.klpo_beta = args.klpo_beta
        self.mc_samples = args.mc_samples

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        # KLPO consumes the raw terminal reward, not the group-relative advantage. Stash the (gathered) per-function
        # rewards so _generate_and_score_completions can recover the raw reward for the local slice.
        self._rewards_per_func = rewards_per_func
        return rewards_per_func

    @torch.no_grad()
    def _sample_mc_records(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draw the MC-KL auxiliary records from the sampler policy (the current model, right after generation).

        At every visited prefix, draws `M = self.mc_samples` token IDs IID with replacement from the model's
        temperature-scaled distribution, independently of the rollout, and records their sampler log-probabilities.
        Also records the sampler log-probabilities of the generated (action) tokens.

        Returns:
            `tuple` of (`mc_ids` of shape `(B, T, M)`, `behavior_mc_logps` of shape `(B, T, M)`,
            `behavior_per_token_logps` of shape `(B, T)`).
        """
        mode = "train" if self.model.training else "eval"
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        all_mc_ids, all_mc_logps, all_action_logps = [], [], []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "use_cache": False}
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            logits = self.model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :][:, -logits_to_keep:, :]
            completion_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # Loop over rows to reduce peak memory (the float32 log-softmax materializes a (T, V) tensor per row)
            for row_logits, row_ids in zip(logits, completion_ids_batch, strict=True):
                row_logps = F.log_softmax(row_logits.float() / self.temperature, dim=-1)  # (T, V)
                all_action_logps.append(row_logps.gather(-1, row_ids.unsqueeze(-1)).squeeze(-1))
                # IID draws WITH replacement; duplicates are valid and must not be merged
                row_mc_ids = torch.multinomial(row_logps.exp(), self.mc_samples, replacement=True)  # (T, M)
                all_mc_ids.append(row_mc_ids)
                all_mc_logps.append(row_logps.gather(-1, row_mc_ids))
        mc_ids = torch.stack(all_mc_ids)
        behavior_mc_logps = torch.stack(all_mc_logps)
        behavior_per_token_logps = torch.stack(all_action_logps)
        return mc_ids, behavior_mc_logps, behavior_per_token_logps

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        output = super()._generate_and_score_completions(inputs)

        if "pixel_values" in output:
            raise NotImplementedError("KLPOTrainer does not support vision-language models yet.")

        device = self.accelerator.device
        # Recover the raw terminal rewards for the local slice (self._rewards_per_func is gathered across processes).
        # Unscorable completions (every reward func returned None) are NaN; zero them so they carry no signal.
        rewards = (self._rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        rewards = torch.nan_to_num(rewards, nan=0.0)
        batch_size = output["completion_ids"].size(0)
        process_slice = slice(
            self.accelerator.process_index * batch_size, (self.accelerator.process_index + 1) * batch_size
        )
        output["raw_rewards"] = rewards[process_slice]

        # Draw the MC-KL auxiliary records from the sampler (the current model, before any update on this batch).
        # These records stay fixed while the batch is reused (num_iterations > 1 / gradient accumulation).
        prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
        completion_ids, completion_mask = output["completion_ids"], output["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mc_ids, behavior_mc_logps, behavior_per_token_logps = self._sample_mc_records(
            input_ids, attention_mask, logits_to_keep
        )
        output["mc_ids"] = mc_ids
        output["behavior_mc_logps"] = behavior_mc_logps
        # `old_per_token_logps` is only computed by GRPO in some configurations; KLPO always needs the sampler's
        # action log-probabilities for the feedback coefficient, so fill it in when absent.
        if "old_per_token_logps" not in output:
            output["old_per_token_logps"] = behavior_per_token_logps
        return output

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Single forward pass: gather the current policy's log-probabilities at the generated (action) tokens and at
        # the M auxiliary MC draws in one go.
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        # Exclude the last value: it corresponds to the next token pred
        logits = logits[:, :-1, :][:, -logits_to_keep:, :]
        mc_ids = inputs["mc_ids"]  # (B, T, M)
        index = torch.cat([completion_ids.unsqueeze(-1), mc_ids], dim=-1)  # (B, T, 1 + M)
        gathered_logps = selective_log_softmax(logits, index, temperature=self.temperature)
        per_token_logps = gathered_logps[..., 0]  # (B, T)
        mc_per_token_logps = gathered_logps[..., 1:]  # (B, T, M)

        raw_rewards = inputs["raw_rewards"]  # (B,)
        old_per_token_logps = inputs["old_per_token_logps"]  # sampler action logps, always set at scoring time
        behavior_mc_logps = inputs["behavior_mc_logps"]  # (B, T, M)

        # Per-token feedback coefficient h = R - beta * ell, with ell = log p(a) - log q(a). Detached.
        ell = per_token_logps.detach() - old_per_token_logps
        h = raw_rewards.unsqueeze(1) - self.klpo_beta * ell  # (B, T)

        # Score-centered correction z = log p(a) - mean_j log p(v_j) (MC-KL estimate of the conditional mean score)
        correction = per_token_logps - mc_per_token_logps.mean(-1)
        per_token_loss = -h * correction

        # Sum over generated tokens WITHOUT length normalization, average over complete responses
        mode = "train" if self.model.training else "eval"
        loss = (per_token_loss * mask).sum(-1).mean()
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        loss = loss / normalizer

        # Log the metrics
        def global_masked_mean(x):
            local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        # Unbiased MC estimate of the local KL(q || p) at each prefix: mean_j (log q(v_j) - log p(v_j)). Individual
        # estimates may be negative; do not clamp them.
        mc_kl = (behavior_mc_logps - mc_per_token_logps.detach()).mean(-1)
        self._metrics[mode]["kl"].append(global_masked_mean(mc_kl))
        self._metrics[mode]["klpo/return_coefficient"].append(global_masked_mean(h))

        return loss
