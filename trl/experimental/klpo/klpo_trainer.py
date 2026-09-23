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

import math
from typing import Any

import torch
import torch.nn.functional as F

from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id, selective_log_softmax
from .klpo_config import KLPOConfig


# Both tail masses of the TopK-KL estimator are clamped to this floor to avoid a 0/0 tail ratio and log(0).
TAIL_FLOOR = 1e-6


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x < 0, without subtracting a rounded probability."""
    return torch.where(x < -math.log(2), torch.log1p(-x.exp()), torch.log(-torch.expm1(x)))


class KLPOTrainer(GRPOTrainer):
    """
    Trainer for KL-Regularized Policy Optimization (KLPO).

    KLPO (https://yifanzhang-pro.github.io/KLPO/) is a critic-free, single-rollout method for off-policy reinforcement
    learning. One complete response per prompt is sufficient; there is no same-prompt response group, learned
    value/normalizer model, importance-ratio multiplier, reward centering, or ratio clipping.

    Let `p` be the current policy, `q` the sampler (behavior) policy, `R` the terminal reward, and `ell_u = log p(a_u)
    - log q(a_u)` for each generated token `a_u`. Two regression routes are supported (`klpo_route`), each combinable
    with four conditional-KL estimators (`kl_estimator`):

    - **Token regression** (default) uses the detached per-token feedback `h_u = R - klpo_beta * ell_u` and a
      score-centered correction `z_u`; the loss is `-mean_responses sum_tokens sg(h_u) * z_u`.
    - **Sequence regression** uses the detached trajectory residual `D = R - klpo_beta * sum_tokens(ell_u + k_u)` (with
      `k_u` the estimated local KL) weighting the summed corrected scores.

    The estimators differ in how the sampler-conditioned score correction (and `k_u`) is computed:

    - `"mc"` (default): `M = mc_samples` independent auxiliary token draws `v_j ~ q` per visited prefix, sampled IID
      with replacement, independently of the rollout. Sequence regression uses leave-one-out residuals (`M >= 2`).
    - `"topk"`: the sampler's `K = kl_top_k` highest-probability tokens per prefix, with all remaining tokens
      aggregated into one tail bucket.
    - `"binary"`: the sampled action against its complement; needs only the sampled-action log-probabilities.
    - `"full"`: the exact KL over the entire vocabulary (stores the sampler's full `(B, T, V)` conditionals).

    In every case, tokens are summed without length normalization and averaged over complete responses. The sampler `q`
    is the training policy snapshot at generation time: right after generation the trainer scores the completions with
    the current model, extracts the estimator's records from that distribution, and keeps them (and the action
    log-probabilities) fixed while the batch is reused. Following the report, no vLLM importance-sampling multiplier is
    applied to the loss, and the group-relative `advantages` computed by [`GRPOTrainer`] are ignored in favor of the
    raw terminal reward.

    The changes w.r.t. [`GRPOTrainer`] are `_compute_loss`, plus a `_generate_and_score_completions` override that
    records the raw rewards and the sampler records. Everything else (generation, reward computation, weight syncing,
    metric logging) is inherited unchanged.
    """

    _tag_names = ["trl", "klpo"]

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = KLPOConfig(f"{model_name.split('/')[-1]}-KLPO")

        super().__init__(model, reward_funcs, args=args, **kwargs)

        self.klpo_route = args.klpo_route
        self.kl_estimator = args.kl_estimator
        self.klpo_beta = args.klpo_beta
        self.mc_samples = args.mc_samples
        self.kl_top_k = args.kl_top_k

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        # KLPO consumes the raw terminal reward, not the group-relative advantage. Stash the (gathered) per-function
        # rewards so _generate_and_score_completions can recover the raw reward for the local slice.
        self._rewards_per_func = rewards_per_func
        return rewards_per_func

    @torch.no_grad()
    def _record_sampler_data(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int
    ) -> dict[str, torch.Tensor]:
        """
        Extract the sampler records from the sampler policy (the current model, right after generation).

        Always records the sampler log-probabilities of the generated (action) tokens (`"behavior_per_token_logps"`,
        shape `(B, T)`). Depending on `kl_estimator`, also records:

        - `"mc"`: `M = mc_samples` token IDs per visited prefix, drawn IID with replacement from the model's
          temperature-scaled distribution independently of the rollout, and their sampler log-probabilities (`"mc_ids"`
          and `"behavior_mc_logps"`, shape `(B, T, M)`).
        - `"topk"`: the sampler's `K = kl_top_k` highest-probability token IDs per prefix and their sampler
          log-probabilities (`"head_ids"` and `"behavior_head_logps"`, shape `(B, T, K)`).
        - `"full"`: the sampler's full conditionals (`"behavior_full_logps"`, shape `(B, T, V)`).
        - `"binary"`: nothing extra (only the sampled-action log-probabilities are needed).
        """
        mode = "train" if self.model.training else "eval"
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        all_action_logps, all_record_ids, all_record_logps, all_full_logps = [], [], [], []
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
                if self.kl_estimator == "mc":
                    # IID draws WITH replacement; duplicates are valid and must not be merged
                    row_record_ids = torch.multinomial(row_logps.exp(), self.mc_samples, replacement=True)  # (T, M)
                    all_record_ids.append(row_record_ids)
                    all_record_logps.append(row_logps.gather(-1, row_record_ids))
                elif self.kl_estimator == "topk":
                    # Keep the sampler's K highest-probability tokens; never renormalize the head or insert the
                    # sampled action
                    row_record_logps, row_record_ids = torch.topk(row_logps, self.kl_top_k, dim=-1)  # (T, K)
                    all_record_ids.append(row_record_ids)
                    all_record_logps.append(row_record_logps)
                elif self.kl_estimator == "full":
                    all_full_logps.append(row_logps)
        records = {"behavior_per_token_logps": torch.stack(all_action_logps)}
        if self.kl_estimator == "mc":
            records["mc_ids"] = torch.stack(all_record_ids)
            records["behavior_mc_logps"] = torch.stack(all_record_logps)
        elif self.kl_estimator == "topk":
            records["head_ids"] = torch.stack(all_record_ids)
            records["behavior_head_logps"] = torch.stack(all_record_logps)
        elif self.kl_estimator == "full":
            records["behavior_full_logps"] = torch.stack(all_full_logps)
        return records

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

        # Extract the sampler records (the current model, before any update on this batch). These records stay fixed
        # while the batch is reused (num_iterations > 1 / gradient accumulation).
        prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
        completion_ids, completion_mask = output["completion_ids"], output["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        records = self._record_sampler_data(input_ids, attention_mask, logits_to_keep)
        behavior_per_token_logps = records.pop("behavior_per_token_logps")
        output.update(records)
        # `old_per_token_logps` is only computed by GRPO in some configurations; KLPO always needs the sampler's
        # action log-probabilities for the feedback coefficient, so fill it in when absent.
        if "old_per_token_logps" not in output:
            output["old_per_token_logps"] = behavior_per_token_logps
        return output

    @staticmethod
    def _binary_correction(
        per_token_logps: torch.Tensor, behavior_per_token_logps: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Binary-KL terms for the sampled action against its complement. Returns the detached correction `omega = (1 - q)
        / (1 - p)` and the detached local Binary KL, both of shape `(B, T)` and zero at masked positions. Requires
        strictly negative active log-probabilities (0 < p, q < 1).
        """
        with torch.no_grad():
            active = mask.bool()
            if (per_token_logps[active] == 0).any() or (behavior_per_token_logps[active] == 0).any():
                raise ValueError("binary KL needs strictly negative logps; compute log_softmax in float32 or float64")
            # Safe non-boundary padding before log1mexp, division, or exponentiation.
            log_p = per_token_logps.detach().float().masked_fill(~active, -1.0)
            log_q = behavior_per_token_logps.float().masked_fill(~active, -1.0)
            log_pc, log_qc = _log1mexp(log_p), _log1mexp(log_q)
            ell = log_p - log_q
            q_complement = -torch.expm1(log_q)
            binary_kl = -log_q.exp() * ell + q_complement * (log_qc - log_pc)
            correction = (log_qc - log_pc).exp()
            return correction.masked_fill(~active, 0.0), binary_kl.masked_fill(~active, 0.0)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Single forward pass: gather the current policy's log-probabilities at the generated (action) tokens and at
        # the estimator's record token IDs in one go.
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        # Exclude the last value: it corresponds to the next token pred
        logits = logits[:, :-1, :][:, -logits_to_keep:, :]

        record_p_logps = full_p_logps = None
        if self.kl_estimator in ("mc", "topk"):
            record_ids = inputs["mc_ids"] if self.kl_estimator == "mc" else inputs["head_ids"]  # (B, T, M or K)
            index = torch.cat([completion_ids.unsqueeze(-1), record_ids], dim=-1)  # (B, T, 1 + M or K)
            gathered_logps = selective_log_softmax(logits, index, temperature=self.temperature)
            per_token_logps = gathered_logps[..., 0]  # (B, T)
            record_p_logps = gathered_logps[..., 1:]  # (B, T, M or K)
        elif self.kl_estimator == "binary":
            per_token_logps = selective_log_softmax(logits, completion_ids, temperature=self.temperature)
        elif self.kl_estimator == "full":
            # The full estimator needs the current policy's entire conditionals, differentiably
            full_p_logps = F.log_softmax(logits.float() / self.temperature, dim=-1)  # (B, T, V)
            per_token_logps = full_p_logps.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

        raw_rewards = inputs["raw_rewards"]  # (B,)
        old_per_token_logps = inputs["old_per_token_logps"]  # sampler action logps, always set at scoring time
        # ell = log p(a) - log q(a); zero at masked positions so trajectory sums ignore padding
        ell = (per_token_logps.detach() - old_per_token_logps) * mask

        # Per-estimator terms:
        # - `correction`: the differentiable corrected score z_u used by token regression (binary folds its omega
        #   multiplier into z_u = omega * log p(a) instead of centering)
        # - `local_kl`: the estimated local KL k_u used by sequence regression (differentiable through the current
        #   policy for topk/full, detached for mc/binary) and for the `kl` metric
        if self.kl_estimator == "mc":
            behavior_mc_logps = inputs["behavior_mc_logps"]  # (B, T, M)
            # Score-centered correction z = log p(a) - mean_j log p(v_j)
            correction = per_token_logps - record_p_logps.mean(-1)
            # Unbiased MC estimate of the local KL(q || p): mean_j (log q(v_j) - log p(v_j)). Individual estimates
            # may be negative; do not clamp them.
            local_kl = (behavior_mc_logps - record_p_logps.detach()).mean(-1) * mask
        elif self.kl_estimator == "binary":
            omega, local_kl = self._binary_correction(per_token_logps, old_per_token_logps, mask)
            # grad(log p(a) + k_bin) = omega * grad(log p(a)); there is no additive centering term
            correction = omega * per_token_logps
        elif self.kl_estimator in ("topk", "full"):
            if self.kl_estimator == "topk":
                q_logps = inputs["behavior_head_logps"]  # (B, T, K)
            else:
                q_logps = inputs["behavior_full_logps"]  # (B, T, V)
            q = q_logps.exp()
            # Convention 0 log 0 = 0; -inf may encode zero sampler support
            safe_q_logps = q_logps.masked_fill(torch.isneginf(q_logps), 0.0)
            p_logps = record_p_logps if self.kl_estimator == "topk" else full_p_logps
            local_kl = (q * (safe_q_logps - p_logps)).sum(-1)
            if self.kl_estimator == "topk":
                # Aggregate all non-head tokens into one tail bucket; both tail masses are floored to avoid a 0/0
                # ratio and log(0)
                sampler_tail = (1 - q.sum(-1)).clamp_min(TAIL_FLOOR)
                trainer_tail = (1 - p_logps.exp().sum(-1)).clamp_min(TAIL_FLOOR)
                local_kl = local_kl + sampler_tail * (sampler_tail.log() - trainer_tail.log())
            local_kl = local_kl * mask
            if self.klpo_route == "token":
                # Token regression detaches the head coefficients: with unfloored tails,
                # grad(k_K) = -grad(sum_head sg(q_v - rho * p_v) * log p(v)); full KL is the K = V limit (no tail)
                with torch.no_grad():
                    if self.kl_estimator == "topk":
                        rho = sampler_tail / trainer_tail
                        coefficients = q - rho.unsqueeze(-1) * p_logps.exp()
                    else:
                        coefficients = q
                correction = per_token_logps - (coefficients * p_logps).sum(-1)

        mode = "train" if self.model.training else "eval"
        if self.klpo_route == "token":
            # Detached per-token feedback h = R - beta * ell
            h = (raw_rewards.unsqueeze(1) - self.klpo_beta * ell).detach()
            per_token_loss = -h * correction
            # Sum over generated tokens WITHOUT length normalization, average over complete responses
            loss = (per_token_loss * mask).sum(-1).mean()
        elif self.klpo_route == "sequence":
            if self.kl_estimator == "mc":
                # Leave-one-out residuals: weight column j's corrected score by the mean D over OTHER columns, which
                # removes the covariance bias of reusing an MC estimate inside both a residual and its derivative
                m = record_p_logps.size(-1)
                with torch.no_grad():
                    ratio = (behavior_mc_logps - record_p_logps) * mask.unsqueeze(-1)  # (B, T, M)
                    residuals = raw_rewards.unsqueeze(1) - self.klpo_beta * (
                        ell.sum(-1, keepdim=True) + ratio.sum(1)
                    )  # (B, M)
                    residual = residuals.mean(-1)  # (B,)
                    other_residuals = residual.unsqueeze(1) - (residuals - residual.unsqueeze(1)) / (m - 1)
                corrected = (per_token_logps * mask).sum(-1, keepdim=True) - (record_p_logps * mask.unsqueeze(-1)).sum(
                    1
                )  # (B, M)
                loss = -(other_residuals * corrected).mean(-1).mean()
            else:
                # Trajectory residual D = R - beta * sum(ell + k), recomputed and detached at each learner step. For
                # topk/full, k is differentiated through the current policy's log-probabilities (including the
                # floored trainer tail); for binary, grad flows through omega * log p(a) only.
                residual = (raw_rewards - self.klpo_beta * (ell + local_kl).sum(-1)).detach()
                if self.kl_estimator == "binary":
                    corrected = (omega * per_token_logps * mask).sum(-1)
                else:  # topk, full
                    corrected = ((per_token_logps + local_kl) * mask).sum(-1)
                loss = -(residual * corrected).mean()
            self._metrics[mode]["klpo/residual"].append(self.accelerator.gather(residual).mean().item())

        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        loss = loss / normalizer

        # Log the metrics
        def global_masked_mean(x):
            local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        self._metrics[mode]["kl"].append(global_masked_mean(local_kl.detach()))
        if self.klpo_route == "token":
            self._metrics[mode]["klpo/return_coefficient"].append(global_masked_mean(h))

        return loss
