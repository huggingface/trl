import gc
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback

from ..models.modeling_base import GeometricMixtureWrapper
from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    truncate_response,
)
from .nash_md_config import NashMDConfig
from .online_dpo_trainer import OnlineDPOTrainer


INVALID_LOGPROB = 1.0


@staticmethod
def logits_to_logprobs(logits, response_ids, temperature=1.0):
    logits /= temperature + 1e-7
    logprobs = F.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, -1, response_ids.unsqueeze(-1)).squeeze(-1)


class NashMDTrainer(OnlineDPOTrainer):
    def __init__(
        self,
        config: NashMDConfig,
        tokenizer: PreTrainedTokenizer,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.model = model

        self.model.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.model.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_model = ref_model
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_updates = args.total_episodes // args.batch_size
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size,
            args.num_generation_per_prompt,
            "`local_batch_size` must be a multiple of `num_generation_per_prompt`",
        )  # DPO logic: repeats the same prompt args.rloo_k times

        ### hyperparams stuff
        self.beta = config.beta
        self.loss_type = config.loss_type

        #########
        # setup model, optimizer, and others
        #########
        if args.disable_dropout:
            disable_dropout_in_model(self.model)
        self.reward_model.eval()

        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.ref_model = prepare_deepspeed(self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16)
            self.deepspeed = self.model
        else:
            self.ref_model = self.ref_model.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model

        reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        ref_model = self.ref_model
        # Mixture model is needed only to generate responses
        mixture_model = GeometricMixtureWrapper(
            model=model,
            ref_model=self.ref_model,
            generation_config=generation_config,
            mixture_coeff=args.mixture_coeff,
            device=accelerator.device,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        loss_stats = torch.zeros(stats_shape, device=device)
        preference_loss_stats = torch.zeros(stats_shape, device=device)
        kl_stats = torch.zeros(stats_shape, device=device)
        preference_model_vs_mixture_stats = torch.zeros(stats_shape, device=device)
        model_logprobs_stats = torch.zeros(stats_shape, device=device)
        ref_logprobs_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_updates * args.num_mini_batches
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

        for update in range(1, args.num_updates + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)

            with torch.no_grad():
                all_queries = []
                all_responses = []
                all_query_responses = []
                all_postprocessed_query_responses = []
                all_sequence_lengths = []
                all_model_rewards = []  # rewards of the responses generated by model
                all_mixture_rewards = []  # rewards of the responses generated by ref mixture model
                all_preference_model_vs_mixture = []  # prefrence of the responses generated by model vs by ref mixture model
                all_ref_logprobs_model_response = []  # logprobs of model responses generated by ref mixture model

                queries = data["input_ids"]
                all_queries.append(queries)
                context_length = queries.shape[1]

                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    query_responses, logits_responses = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )

                with unwrap_model_for_generation(mixture_model, self.accelerator) as unwrapped_mixture_model:
                    mixture_query_responses, mixture_logits_responses = batch_generation(
                        unwrapped_mixture_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )

                # To have a generality of the code
                all_query_responses.append(query_responses)

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    # get responses from the model and the reference mixture model
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    mixture_query_response = mixture_query_responses[i : i + args.local_rollout_forward_batch_size]
                    mixture_response = mixture_query_response[:, context_length:]

                    all_query_responses.append(query_response)
                    all_responses.append(response)

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    postprocessed_mixture_response = mixture_response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )
                        postprocessed_mixture_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, mixture_response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    postprocessed_mixture_query_response = torch.cat((query, postprocessed_mixture_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    all_postprocessed_query_responses.append(postprocessed_query_response)
                    all_sequence_lengths.append(sequence_length)

                    # Compute rewards for both sets of responses
                    _, model_reward, _ = get_reward(
                        reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                    )
                    _, mixture_reward, _ = get_reward(
                        reward_model, postprocessed_mixture_query_response, tokenizer.pad_token_id, context_length
                    )
                    # if the responses do not contain eos token then set the reward to penalty_reward_value
                    if args.non_eos_penalty:
                        model_resp_contains_eos = torch.any(
                            postprocessed_query_response == tokenizer.eos_token_id, dim=-1
                        )
                        model_reward = torch.where(model_resp_contains_eos, model_reward, args.penalty_reward_value)
                        ref_resp_contains_eos = torch.any(
                            postprocessed_mixture_query_response == tokenizer.eos_token_id, dim=-1
                        )
                        mixture_reward = torch.where(ref_resp_contains_eos, mixture_reward, args.penalty_reward_value)

                    # Save model rewards for logging
                    all_model_rewards.append(model_reward)
                    all_mixture_rewards.append(mixture_reward)

                    # Compute the preference between the model and the mixture model
                    # TODO: Replace it by a soft judge instead of BT model (P(model > ref) = sigmoid(model_reward - ref_reward))
                    preference_model_vs_mixture = F.sigmoid(model_reward - mixture_reward)
                    all_preference_model_vs_mixture.append(preference_model_vs_mixture)

                    # Compute the logprobs of the responses generated by the model, preserve all the disribution; shape [batch_size, response_length, vocab_size]
                    ref_logprobs_model_response = self.compute_logprobs(ref_model, query_response, context_length)
                    all_ref_logprobs_model_response.append(ref_logprobs_model_response)

                # stack all the tensors
                all_queries = torch.cat(all_queries, dim=0)
                all_responses = torch.cat(all_responses, dim=0)
                all_query_responses = torch.cat(all_query_responses, dim=0)
                all_postprocessed_query_responses = torch.cat(all_postprocessed_query_responses, dim=0)
                all_sequence_lengths = torch.cat(all_sequence_lengths, dim=0)
                all_model_rewards = torch.cat(all_model_rewards, dim=0)
                all_mixture_rewards = torch.cat(all_mixture_rewards, dim=0)
                all_preference_model_vs_mixture = torch.cat(all_preference_model_vs_mixture, dim=0)
                all_ref_logprobs_model_response = torch.cat(all_ref_logprobs_model_response, dim=0)

                del model_reward, mixture_reward, preference_model_vs_mixture, ref_logprobs_model_response
                torch.cuda.empty_cache()
                gc.collect()

            # Do multiple epochs of training, with a fresh random shuffle in each epoch
            for epoch_idx in range(args.num_epochs):
                b_inds = np.random.permutation(args.local_batch_size // args.num_generation_per_prompt)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0,
                    args.local_batch_size // args.num_generation_per_prompt,
                    args.local_mini_batch_size // args.num_generation_per_prompt,
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size // args.num_generation_per_prompt
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                        0,
                        args.local_mini_batch_size // args.num_generation_per_prompt,
                        args.per_device_train_batch_size,
                    ):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            ## context lengths
                            context_lengths = all_queries[micro_batch_inds].shape[1]

                            ## response, query-responses and sequence lengths
                            responses = all_responses[micro_batch_inds]
                            query_responses = all_query_responses[micro_batch_inds]
                            sequence_lengths = all_sequence_lengths[micro_batch_inds]

                            ## padding masks
                            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(
                                responses.shape[0], 1
                            )
                            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)

                            ## compute model log_probs, shape [batch_size, response_length, vocab_size]
                            model_logprobs = self.compute_logprobs(model, query_responses, context_lengths)

                            ## pre-calcualted preference model vs mixture
                            preference_model_vs_mixture = all_preference_model_vs_mixture[micro_batch_inds]

                            ## pre-calculated ref logprobs for model response
                            ref_logprobs = all_ref_logprobs_model_response[micro_batch_inds]

                            ## Preference loss (sign `-` is because we want to maximize the preference instead of minimizing)
                            model_logprobs_over_gen = torch.gather(
                                model_logprobs, 2, query_responses[:, context_lengths:].unsqueeze(-1)
                            ).squeeze(-1)
                            model_logprobs_sum = torch.sum(model_logprobs_over_gen * ~padding_mask, dim=1)
                            preference_losses = -model_logprobs_sum * (
                                preference_model_vs_mixture - 0.5
                            )  # 0.5 is a control variate

                            ## KL loss
                            raw_kl_model_vs_ref = torch.sum(
                                torch.exp(model_logprobs) * (model_logprobs - ref_logprobs), dim=2
                            )
                            kl_model_vs_ref = torch.sum(raw_kl_model_vs_ref * ~padding_mask, dim=1)

                            # total loss
                            loss = (preference_losses + self.beta * kl_model_vs_ref).mean()

                            ## ref logprobs for logging
                            ref_logprobs_over_gen = torch.gather(
                                ref_logprobs, 2, query_responses[:, context_lengths:].unsqueeze(-1)
                            ).squeeze(-1)
                            ref_logprobs_sum = torch.sum(ref_logprobs_over_gen * ~padding_mask, dim=1)

                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss
                                preference_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    preference_losses.mean()
                                )
                                kl_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = kl_model_vs_ref.mean()
                                preference_model_vs_mixture_stats[
                                    epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = preference_model_vs_mixture.mean()
                                model_logprobs_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    model_logprobs_sum.mean()
                                )
                                ref_logprobs_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    ref_logprobs_sum.mean()
                                )
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1

            with torch.no_grad():
                model_reward = all_model_rewards.mean()
                mixture_reward = all_mixture_rewards.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(kl_stats).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(-model_logprobs_stats).mean().item()
                metrics["objective/preference"] = (
                    self.accelerator.gather(preference_model_vs_mixture_stats).mean().item()
                )
                metrics["objective/model_rew"] = self.accelerator.gather(model_reward).mean().item()
                metrics["objective/mixture_rew"] = self.accelerator.gather(mixture_reward).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(loss_stats).mean().item()
                metrics["loss/pref_loss_avg"] = self.accelerator.gather(preference_loss_stats).mean().item()
                metrics["logps/model"] = self.accelerator.gather(model_logprobs_stats).mean().item()
                metrics["logps/ref"] = self.accelerator.gather(ref_logprobs_stats).mean().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.log(metrics)

            self.lr_scheduler.step()
            self.state.global_step += 1

            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def compute_logprobs(self, model, responses, context_length, ref_model=None, mixture_coeff=0):
        output = forward(model, responses, self.tokenizer.pad_token_id)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= self.args.temperature + 1e-7

        if ref_model is not None:
            with torch.no_grad():
                ref_output = forward(ref_model, responses, self.tokenizer.pad_token_id)
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                ref_logits /= self.args.temperature + 1e-7
                logits = mixture_coeff * ref_logits + (1 - mixture_coeff) * logits

        # return the log of the distribution over tokens
        return F.log_softmax(logits, dim=-1)  # shape [batch_size, response_length, vocab_size]
