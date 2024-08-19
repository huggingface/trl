import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from ..models.utils import unwrap_model_for_generation
from .online_dpo_trainer import OnlineDPOTrainer
from .utils import (
    batch_generation,
    forward,
    get_reward,
)


@staticmethod
def logits_to_logprobs(logits, response_ids, temperature=1.0):
    logits /= temperature + 1e-7
    logprobs = F.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, -1, response_ids.unsqueeze(-1)).squeeze(-1)


class XPOTrainer(OnlineDPOTrainer):
    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_model = self.ref_model
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from self.dataloader

        iter_dataloader = iter(repeat_generator())

        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        start_time = time.time()
        stats_shape = (args.num_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        loss_stats = torch.zeros(stats_shape, device=device)
        chosen_rewards_stats = torch.zeros(stats_shape, device=device)
        rejected_rewards_stats = torch.zeros(stats_shape, device=device)
        chosen_logprobs_stats = torch.zeros(stats_shape, device=device)
        rejected_logprobs_stats = torch.zeros(stats_shape, device=device)
        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
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

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            ref_model.eval()
            model.eval()

            # data collation
            with torch.no_grad():
                all_queries = []
                all_chosen_responses = []
                all_rejected_responses = []
                all_ref_logprobs_chosen = []
                all_ref_logprobs_rejected = []
                all_ref_responses = []
                all_reward_margins = []
                queries = data["input_ids"]
                all_queries.append(queries)

                context_length = queries.shape[1]
                # model responses
                with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                    model_responses, model_logits = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )

                # reference model responses
                with unwrap_model_for_generation(ref_model, accelerator) as unwrapped_model:
                    ref_responses, ref_logits = batch_generation(
                        ref_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )
                    all_ref_responses.append(ref_responses)

                # compute rewards and ref log probs in local rollout forward batch sizes:
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    model_response = model_responses[i : i + args.local_rollout_forward_batch_size]
                    ref_response = ref_responses[i : i + args.local_rollout_forward_batch_size]

                    # Compute rewards for both sets of responses
                    _, model_reward, _ = get_reward(
                        reward_model, model_response, tokenizer.pad_token_id, context_length
                    )
                    _, ref_reward, _ = get_reward(reward_model, ref_response, tokenizer.pad_token_id, context_length)
                    # if the responses do not contain eos token then set the reward to penalty_reward_value
                    if args.non_eos_penalty:
                        model_resp_contains_eos = torch.any(model_response == tokenizer.eos_token_id, dim=-1)
                        model_reward = torch.where(model_resp_contains_eos, model_reward, args.penalty_reward_value)
                        ref_resp_contains_eos = torch.any(ref_response == tokenizer.eos_token_id, dim=-1)
                        ref_reward = torch.where(ref_resp_contains_eos, ref_reward, args.penalty_reward_value)

                    # Create preference dataset based on rewards
                    chosen_mask = model_reward >= ref_reward
                    rejected_mask = ~chosen_mask

                    # reward margin between chosen and rejected responses
                    reward_margin = torch.where(chosen_mask, model_reward - ref_reward, ref_reward - model_reward)
                    all_reward_margins.append(reward_margin)

                    # chosen and rejected responses
                    chosen_response = torch.where(chosen_mask.unsqueeze(1), model_response, ref_response)
                    rejected_response = torch.where(rejected_mask.unsqueeze(1), model_response, ref_response)
                    all_chosen_responses.append(chosen_response)
                    all_rejected_responses.append(rejected_response)

                    ref_logprobs_model_response = self.compute_logprobs(ref_model, model_response, context_length)
                    ref_logprobs_ref_response = self.compute_logprobs(ref_model, ref_response, context_length)

                    ref_logprobs_chosen = torch.where(
                        chosen_mask.unsqueeze(1), ref_logprobs_model_response, ref_logprobs_ref_response
                    )
                    ref_logprobs_rejected = torch.where(
                        rejected_mask.unsqueeze(1), ref_logprobs_model_response, ref_logprobs_ref_response
                    )

                    all_ref_logprobs_chosen.append(ref_logprobs_chosen)
                    all_ref_logprobs_rejected.append(ref_logprobs_rejected)

                # stack all the tensors
                all_queries = torch.cat(all_queries, dim=0)
                all_chosen_responses = torch.cat(all_chosen_responses, dim=0)
                all_rejected_responses = torch.cat(all_rejected_responses, dim=0)
                all_ref_logprobs_chosen = torch.cat(all_ref_logprobs_chosen, dim=0)
                all_ref_logprobs_rejected = torch.cat(all_ref_logprobs_rejected, dim=0)
                all_ref_responses = torch.cat(all_ref_responses, dim=0)
                all_reward_margins = torch.cat(all_reward_margins, dim=0)
                torch.cuda.empty_cache()

            # logging kl between the model and ref model responses
            logprobs_model = logits_to_logprobs(model_logits, model_responses[:, context_length:], args.temperature)
            logprobs_ref = logits_to_logprobs(ref_logits, ref_responses[:, context_length:], args.temperature)
            kl = logprobs_model - logprobs_ref
            dpo_score = -args.beta * (kl).sum(1).mean()
            scores_margin = all_reward_margins.mean()

            # Do multiple epochs of XPO training, with a fresh random shuffle in each epoch
            model.train()
            for epoch_idx in range(args.num_epochs):
                b_inds = np.random.permutation(args.local_batch_size // self.num_generation_per_prompt)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0,
                    args.local_batch_size // self.num_generation_per_prompt,
                    args.local_mini_batch_size // self.num_generation_per_prompt,
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size // self.num_generation_per_prompt
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                        0,
                        args.local_mini_batch_size // self.num_generation_per_prompt,
                        args.per_device_train_batch_size,
                    ):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            ## context lengths
                            context_lengths = all_queries[micro_batch_inds].shape[1]

                            ## chosen
                            chosen_responses = all_chosen_responses[micro_batch_inds]
                            ## rejected
                            rejected_responses = all_rejected_responses[micro_batch_inds]

                            ## concated log_probs to compute log probs
                            concated_logprobs = self.compute_logprobs(
                                model,
                                torch.cat([chosen_responses, rejected_responses], dim=0),
                                context_lengths,
                            )

                            # model logprobs
                            (chosen_logprobs, rejected_logprobs) = torch.split(
                                concated_logprobs, [chosen_responses.shape[0], rejected_responses.shape[0]]
                            )

                            # pre-calculated ref logprobs for chosen and rejected
                            ref_logprobs_chosen = all_ref_logprobs_chosen[micro_batch_inds]
                            ref_logprobs_rejected = all_ref_logprobs_rejected[micro_batch_inds]

                            # log ratios
                            chosen_log_ratios = chosen_logprobs.sum(1) - ref_logprobs_chosen.sum(1)
                            rejected_log_ratios = rejected_logprobs.sum(1) - ref_logprobs_rejected.sum(1)
                            diff_log_ratios = chosen_log_ratios - rejected_log_ratios

                            # dpo losses
                            if self.loss_type == "sigmoid":
                                dpo_losses = -F.logsigmoid(self.beta * diff_log_ratios)
                            elif self.loss_type == "ipo":
                                dpo_losses = (diff_log_ratios - 1 / (2 * self.beta)) ** 2
                            else:
                                raise NotImplementedError(f"invalid loss type {self.loss_type}")

                            # xpo losses
                            model_logprobs_ref = self.compute_logprobs(
                                model, all_ref_responses[micro_batch_inds], context_lengths
                            )
                            xpo_losses = args.alpha * model_logprobs_ref.sum(1)

                            # total loss
                            loss = (dpo_losses + xpo_losses).mean()

                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                chosen_rewards = self.beta * chosen_log_ratios.detach()
                                rejected_rewards = self.beta * rejected_log_ratios.detach()
                                xpo_loss = xpo_losses.detach()
                                loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss.detach()
                                chosen_rewards_stats[
                                    epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = chosen_rewards.mean()
                                rejected_rewards_stats[
                                    epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = rejected_rewards.mean()
                                chosen_logprobs_stats[
                                    epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = chosen_log_ratios.mean()
                                rejected_logprobs_stats[
                                    epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = rejected_log_ratios.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    self.state.global_step += 1

            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = -logprobs_model.mean()
                mean_dpo_score = dpo_score.item()
                eps = int(self.state.episode / (time.time() - start_time))
                g_chosen_reward = self.accelerator.gather(chosen_rewards_stats)
                g_rejected_reward = self.accelerator.gather(rejected_rewards_stats)
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/mean_dpo_score"] = self.accelerator.gather(mean_dpo_score)
                metrics["objective/xpo_loss"] = self.accelerator.gather(xpo_loss).mean().item()
                metrics["objective/scores_margin"] = self.accelerator.gather(scores_margin.mean()).mean().item()
                metrics["rewards/chosen"] = g_chosen_reward.mean().item()
                metrics["rewards/rejected"] = g_rejected_reward.mean().item()
                metrics["rewards/accuracies"] = (g_chosen_reward > g_rejected_reward).float().mean().item()
                metrics["rewards/margins"] = (g_chosen_reward - g_rejected_reward).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(loss_stats).mean().item()
                metrics["logps/chosen"] = self.accelerator.gather(chosen_logprobs_stats).mean().item()
                metrics["logps/rejected"] = self.accelerator.gather(rejected_logprobs_stats).mean().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def compute_logprobs(self, model, responses, context_length):
        output = forward(model, responses, self.tokenizer.pad_token_id)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= self.args.temperature + 1e-7
        logprobs = F.log_softmax(logits, dim=-1)
        target_ids = responses[:, context_length:]
        return torch.gather(logprobs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
