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

"""
DPPO (Divergence Proximal Policy Optimization) Trainer

Implements the Stable-RL DPPO algorithm which replaces PPO's clipped surrogate
objective with a divergence-based binary token mask.  Instead of clipping the
probability ratio, DPPO zeroes out gradient contributions from tokens whose
policy probability has moved outside a trust region measured by a direct
divergence metric (Total Variation or binary KL).

Reference: https://github.com/sail-sg/Stable-RL
Paper: "Rethinking the Trust Region in LLM Reinforcement Learning" (arXiv 2602.04879)
"""

import gc
import math
import textwrap
import time

import numpy as np
import torch
from accelerate import logging
from transformers import GenerationConfig

from ..ppo.ppo_trainer import PPOTrainer as BasePPOTrainer
from ..ppo.ppo_trainer import (
    batch_generation,
    forward,
    masked_mean,
    masked_whiten,
    selective_log_softmax,
    truncate_response,
)
from ...models.utils import unwrap_model_for_generation
from ...trainer.utils import empty_cache
from ..utils import first_true_indices, get_reward
from .dppo_config import DPPOConfig


logger = logging.get_logger(__name__)

INVALID_LOGPROB = 1.0


class DPPOTrainer(BasePPOTrainer):
    """Trainer for Divergence Proximal Policy Optimization (DPPO).

    DPPO replaces PPO's clipped surrogate objective with a divergence-based binary
    token mask.  For each response token the trainer computes a divergence between the
    current policy and the rollout policy (either Total Variation or binary KL).  Tokens
    whose divergence exceeds a threshold are masked out (gradient set to zero) rather than
    being soft-clipped as in standard PPO.  Within the trust region the loss is the
    IS-weighted policy gradient:

        L_DPPO = -A · clamp(π_current/π_rollout, max=c) · mask · log π_current

    where ``mask`` is 1 for tokens inside the trust region and 0 for tokens outside.
    The IS weight is detached so the gradient flows entirely through ``log π_current``.

    Key algorithmic differences from PPO:

    * Trust-region metric: divergence on the *sampled* token (probability difference or
      binary KL) rather than the probability ratio.  This is bounded and numerically
      stable for low-probability tokens.
    * Enforcement: hard zeroing via a binary mask rather than soft clipping.
    * Gradient signal: ``log π`` (policy gradient style) rather than the ratio surrogate.

    Three mask variants are supported via ``DPPOConfig.loss_mode``:

    * ``"dppo_binary_tv"``: Total Variation — mask if ``|p_current − p_rollout| > ε``.
    * ``"dppo_binary_kl"``: Binary KL anchored to rollout-engine probabilities.
    * ``"dppo_binary_kl_recompute"``: Binary KL anchored to training-engine probabilities
      recomputed at the start of each PPO epoch (MiniRL-style).

    Args:
        args ([`experimental.dppo.DPPOConfig`]):
            Training arguments.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`]):
            Class to process the data.
        model (`torch.nn.Module`):
            Model to be trained. This is the policy model.
        ref_model (`torch.nn.Module`, *optional*):
            Reference model used to compute the KL divergence. If `None`, a copy of the policy model is created.
        reward_model (`torch.nn.Module`):
            Reward model used to compute the rewards.
        train_dataset ([`~datasets.Dataset`]):
            Dataset for training.
        value_model (`torch.nn.Module`):
            Value model used to predict the value of a state.
        data_collator ([`~transformers.DataCollatorWithPadding`], *optional*):
            Data collator to batch and pad samples from the dataset. If `None`, a default data collator is
            created using the `processing_class`.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*,
            defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training. If `None`,
            the optimizer and the learning rate scheduler are created using the
            [`~transformers.Trainer.create_optimizer_and_scheduler`] method.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the
            policy `model` will be wrapped with the specified PEFT adapter.
    """

    _tag_names = ["trl", "dppo"]
    _name = "DPPO"
    _paper = {
        "title": "Rethinking the Trust Region in LLM Reinforcement Learning",
        "url": "https://arxiv.org/abs/2602.04879",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{dppo2025,
                title        = {{Rethinking the Trust Region in LLM Reinforcement Learning}},
                author       = {SAIL-SG},
                year         = 2025,
                journal      = {arXiv},
                url          = {https://arxiv.org/abs/2602.04879}
            }"""),
    }

    def _compute_dppo_mask(
        self,
        new_logprobs: torch.Tensor,
        rollout_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the DPPO binary token mask.

        For each token, determines whether it is inside or outside the trust region based
        on a direct divergence measure between the current policy and the rollout policy.
        Tokens outside the trust region are masked to zero.

        Args:
            new_logprobs: Current policy log-probabilities ``[batch, seq]``.
            rollout_logprobs: Log-probabilities from the rollout engine ``[batch, seq]``.
            old_logprobs: Log-probabilities from training engine at epoch start ``[batch, seq]``.
                Used only for ``"dppo_binary_kl_recompute"``; pass ``rollout_logprobs`` otherwise.
            advantages: Advantage estimates ``[batch, seq]``.
            response_mask: Boolean mask for valid (non-padding) response tokens ``[batch, seq]``.

        Returns:
            token_mask: Float tensor, shape ``[batch, seq]``, with ``1`` for tokens inside the
                trust region and within ``response_mask``, ``0`` otherwise.
            invalid_mask: Boolean tensor, shape ``[batch, seq]``, ``True`` for masked-out tokens.
        """
        args = self.args
        clip_low = args.clip_ratio_low
        clip_high = args.clip_ratio_high

        prob = torch.exp(new_logprobs)
        rollout_prob = torch.exp(rollout_logprobs)

        if args.loss_mode == "dppo_binary_tv":
            # Total Variation: mask based on |p_current - p_rollout|
            prob_diff = prob - rollout_prob
            invalid_positive_mask = prob_diff > clip_high
            invalid_negative_mask = prob_diff < -clip_low

        elif args.loss_mode == "dppo_binary_kl":
            # Binary KL between Bernoulli(p_rollout) and Bernoulli(p_current),
            # anchored to rollout-engine probabilities.
            kl = rollout_prob * (rollout_logprobs - new_logprobs) + (1.0 - rollout_prob) * torch.log(
                (1.0 - rollout_prob + 1e-8) / (1.0 - prob + 1e-8)
            )
            invalid_positive_mask = (kl > clip_high) & (prob > rollout_prob)
            invalid_negative_mask = (kl > clip_low) & (prob < rollout_prob)

        elif args.loss_mode == "dppo_binary_kl_recompute":
            # Binary KL anchored to training-engine probabilities at epoch start (MiniRL-style).
            old_prob = torch.exp(old_logprobs)
            kl = old_prob * (old_logprobs - new_logprobs) + (1.0 - old_prob) * torch.log(
                (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
            )
            invalid_positive_mask = (kl > clip_high) & (prob > old_prob)
            invalid_negative_mask = (kl > clip_low) & (prob < old_prob)

        else:
            raise ValueError(
                f"Unknown loss_mode: '{args.loss_mode}'. "
                "Expected one of: 'dppo_binary_tv', 'dppo_binary_kl', 'dppo_binary_kl_recompute'."
            )

        # Select the mask based on the sign of the advantage
        invalid_mask = torch.where(advantages > 0, invalid_positive_mask, invalid_negative_mask)

        # Detach: the mask is not differentiated, it acts as a hard gate
        token_mask = (1.0 - invalid_mask.detach().float()) * response_mask.float()
        return token_mask, invalid_mask.detach()

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_kwargs = {
            "max_new_tokens": args.response_length,
            "temperature": (args.temperature + 1e-7),
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
        }
        generation_config = GenerationConfig(**generation_kwargs)

        accelerator.print("===training policy with DPPO===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_maskfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                with (
                    unwrap_model_for_generation(
                        self.model,
                        self.accelerator,
                        gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                        generation_kwargs=generation_kwargs,
                    ) as unwrapped_model
                ):
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
                    else:
                        ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

                # Create padding masks
                # See https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                logr = ref_logprobs - logprobs
                kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[actual_start, actual_end] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                empty_cache()

            # DPPO training: jointly optimise policy and value with the divergence-masked loss.
            # At the start of each PPO epoch we snapshot the training-engine log-probs so that
            # dppo_binary_kl_recompute can use them as the epoch-start anchor.
            epoch_start_logprobs = logprobs  # rollout logprobs used as default anchor

            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)

                if args.loss_mode == "dppo_binary_kl_recompute" and ppo_epoch_idx == 0:
                    # The first epoch anchor equals the rollout logprobs; we recompute below for
                    # subsequent epochs after the first weight update.
                    epoch_start_logprobs = logprobs

                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]          # rollout logprobs
                            mb_old_logprobs = epoch_start_logprobs[micro_batch_inds]  # epoch-start anchor
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]
                            mb_response_mask = ~padding_mask[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )

                            # Value function loss (same as standard PPO)
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])

                            # DPPO policy loss
                            # 1. Compute the binary divergence mask
                            token_mask, invalid_mask = self._compute_dppo_mask(
                                new_logprobs=new_logprobs,
                                rollout_logprobs=mb_logprobs,
                                old_logprobs=mb_old_logprobs,
                                advantages=mb_advantage,
                                response_mask=mb_response_mask,
                            )

                            # 2. IS weight (detached): π_current / π_rollout, capped at clip_ratio_c
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.clamp(torch.exp(logprobs_diff), max=args.clip_ratio_c).detach()

                            # 3. L_DPPO = -A · ratio · mask · log π_current
                            #    Gradient flows through log π_current (new_logprobs) only.
                            pg_losses = -mb_advantage * ratio * token_mask * new_logprobs
                            # Normalise by the number of valid (unmasked) tokens
                            num_valid = token_mask.sum().clamp(min=1.0)
                            pg_loss = pg_losses.sum() / num_valid

                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                # Fraction of valid tokens that were masked out
                                pg_maskfrac = masked_mean(invalid_mask.float(), mb_response_mask)
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_maskfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_maskfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac = masked_mean(
                                    (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                                )
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio,
                        token_mask, invalid_mask, pg_losses, pg_loss, loss,
                        pg_maskfrac, prob_dist, entropy, approxkl,
                        mb_return, mb_advantage, mb_values, mb_responses,
                        mb_query_responses, mb_logprobs, mb_old_logprobs, mb_response_mask,
                    )
                    # fmt: on
                    empty_cache()

                # For dppo_binary_kl_recompute: after the first epoch weight update, re-snapshot
                # the training-engine log-probs as the new epoch-start anchor.
                if args.loss_mode == "dppo_binary_kl_recompute" and ppo_epoch_idx < args.num_ppo_epochs - 1:
                    with torch.no_grad():
                        new_epoch_logprobs_list = []
                        for i in range(0, query_responses.shape[0], args.local_rollout_forward_batch_size):
                            mb_qr = query_responses[i : i + args.local_rollout_forward_batch_size]
                            mb_resp = responses[i : i + args.local_rollout_forward_batch_size]
                            out, _ = forward(model, mb_qr, processing_class.pad_token_id)
                            lgt = out.logits[:, context_length - 1 : -1] / (args.temperature + 1e-7)
                            lp = selective_log_softmax(lgt, mb_resp)
                            new_epoch_logprobs_list.append(lp)
                        epoch_start_logprobs = torch.cat(new_epoch_logprobs_list, 0)
                        epoch_start_logprobs = torch.masked_fill(epoch_start_logprobs, padding_mask, INVALID_LOGPROB)

            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/maskfrac_avg"] = (
                    self.accelerator.gather_for_metrics(pg_maskfrac_stats).mean().item()
                )
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
                epoch_start_logprobs,
            )
            empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
