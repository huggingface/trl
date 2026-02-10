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
DPPO (Decoupled Proximal Policy Optimization) Trainer

This implements the Stable-RL DPPO algorithm which decouples the optimization
of the policy and value function for improved training stability.
Reference: https://github.com/sail-sg/Stable-RL
"""

import gc
import math
import textwrap
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import logging

from ..ppo.ppo_trainer import PPOTrainer as BasePPOTrainer
from ..ppo.ppo_trainer import (
    batch_generation,
    forward,
    masked_mean,
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
    """Trainer for Decoupled Proximal Policy Optimization (DPPO).

    DPPO is a variant of PPO that decouples the optimization of the policy and value function
    for improved training stability. Key features:
    - Separate optimizers and learning rates for policy and value function
    - Independent training loops for policy and value function
    - Configurable value function update frequency

    For details on the Stable-RL approach, see: https://github.com/sail-sg/Stable-RL

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
            Data collator to batch and pad samples from the dataset. If `None`, a default data collator is created
            using the `processing_class`.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training. If `None`, the
            optimizer and the learning rate scheduler are created using the
            [`~transformers.Trainer.create_optimizer_and_scheduler`] method.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the policy `model`
            will be wrapped with the specified PEFT adapter.
    """

    _tag_names = ["trl", "dppo"]
    _name = "DPPO"
    _paper = {
        "title": "Stable-RL: Decoupled Proximal Policy Optimization for Stable Reinforcement Learning",
        "url": "https://github.com/sail-sg/Stable-RL",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @misc{stable-rl,
                title        = {{Stable-RL: Decoupled Proximal Policy Optimization}},
                author       = {SAIL-SG},
                year         = 2023,
                url          = {https://github.com/sail-sg/Stable-RL}
            }"""),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create separate optimizer for value function with its own learning rate
        self.vf_optimizer = self._create_value_optimizer()
        self.vf_lr_scheduler = self._create_value_lr_scheduler()
        
        # Prepare value optimizer with accelerator
        self.vf_optimizer, self.vf_lr_scheduler = self.accelerator.prepare(
            self.vf_optimizer, self.vf_lr_scheduler
        )

    def _create_value_optimizer(self):
        """Create optimizer specifically for the value function."""
        # Get value model parameters
        value_params = self.value_model.parameters()
        
        # Use the value function learning rate from config
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {
            "lr": self.args.vf_learning_rate,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "weight_decay": self.args.weight_decay,
        }
        
        return optimizer_cls(value_params, **optimizer_kwargs)

    def _create_value_lr_scheduler(self):
        """Create learning rate scheduler for the value function optimizer."""
        from transformers.trainer_utils import SchedulerType
        from transformers.optimization import get_scheduler
        
        return get_scheduler(
            SchedulerType(self.args.lr_scheduler_type),
            optimizer=self.vf_optimizer,
            num_warmup_steps=self.args.get_warmup_steps(self.args.num_total_batches),
            num_training_steps=self.args.num_total_batches,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        vf_optimizer = self.vf_optimizer
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
        from transformers import GenerationConfig
        generation_config = GenerationConfig(**generation_kwargs)

        accelerator.print("===training policy with DPPO===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        vf_stats_shape = (args.num_vf_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(vf_stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(vf_stats_shape, device=device)
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
                    from ..ppo.ppo_trainer import masked_whiten
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
                from ..ppo.ppo_trainer import masked_whiten
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                empty_cache()

            # DPPO: Train policy and value function with decoupled optimization
            
            # Phase 1: Policy optimization
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
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
                            mb_logprobs = logprobs[micro_batch_inds]

                            output, _ = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            
                            accelerator.backward(pg_loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    del (
                        output, logits, new_logprobs, logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss_max, pg_loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    empty_cache()

            # Phase 2: Value function optimization (decoupled from policy)
            if update % args.vf_update_frequency == 0:
                for vf_epoch_idx in range(args.num_vf_epochs):
                    b_inds = np.random.permutation(args.local_batch_size)
                    minibatch_idx = 0
                    for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                        mini_batch_end = mini_batch_start + args.local_mini_batch_size
                        mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                        gradient_accumulation_idx = 0
                        for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                            with accelerator.accumulate(model):
                                micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                                micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                                mb_return = returns[micro_batch_inds]
                                mb_values = values[micro_batch_inds]
                                mb_query_responses = query_responses[micro_batch_inds]

                                _, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
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
                                
                                accelerator.backward(vf_loss)
                                vf_optimizer.step()
                                vf_optimizer.zero_grad()
                                
                                with torch.no_grad():
                                    vf_clipfrac = masked_mean(
                                        (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                                    )
                                    vf_loss_stats[vf_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                    vf_clipfrac_stats[vf_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                        vf_clipfrac
                                    )
                            gradient_accumulation_idx += 1
                        minibatch_idx += 1
                        del (
                            vpred_temp, vpred, vpredclipped, vf_losses1, vf_losses2,
                            vf_loss_max, vf_loss, vf_clipfrac, mb_return, mb_values, mb_query_responses,
                        )
                        empty_cache()

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
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["vf_lr"] = self.vf_lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.vf_lr_scheduler.step()
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
            )
            empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
