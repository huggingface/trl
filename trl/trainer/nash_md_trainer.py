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
    forward,
    get_reward,
    prepare_deepspeed,
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
        ref_model = GeometricMixtureWrapper(
            ref_model=self.ref_model,
            model=model,
            generation_config=generation_config,
            mixture_coeff=args.mixture_coeff,
            device=accelerator.device,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        loss_stats = torch.zeros(stats_shape, device=device)
        chosen_rewards_stats = torch.zeros(stats_shape, device=device)
        rejected_rewards_stats = torch.zeros(stats_shape, device=device)
        chosen_logprobs_stats = torch.zeros(stats_shape, device=device)
        rejected_logprobs_stats = torch.zeros(stats_shape, device=device)
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
                all_chosen_responses = []
                all_rejected_responses = []
                all_ref_logprobs_chosen = []
                all_ref_logprobs_rejected = []
                all_ref_responses = []
                all_reward_margins = []
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

                with unwrap_model_for_generation(ref_model, self.accelerator) as unwrapped_ref_model:
                    mixture_query_responses, mixture_logits_responses = batch_generation(
                        unwrapped_ref_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    # get responses from the model and the reference mixture model
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    mixture_query_response = mixture_query_responses[i : i + args.local_rollout_forward_batch_size]

                    # Compute rewards for both sets of responses
                    _, model_reward, _ = get_reward(
                        reward_model, query_response, tokenizer.pad_token_id, context_length
                    )
                    _, ref_reward, _ = get_reward(
                        reward_model, mixture_query_response, tokenizer.pad_token_id, context_length
                    )
                    # if the responses do not contain eos token then set the reward to penalty_reward_value
                    if args.non_eos_penalty:
                        model_resp_contains_eos = torch.any(query_response == tokenizer.eos_token_id, dim=-1)
                        model_reward = torch.where(model_resp_contains_eos, model_reward, args.penalty_reward_value)
                        ref_resp_contains_eos = torch.any(mixture_query_response == tokenizer.eos_token_id, dim=-1)
                        ref_reward = torch.where(ref_resp_contains_eos, ref_reward, args.penalty_reward_value)

                    # Create preference dataset based on rewards
                    chosen_mask = model_reward >= ref_reward
                    rejected_mask = ~chosen_mask

                    # reward margin between chosen and rejected responses
                    reward_margin = torch.where(chosen_mask, model_reward - ref_reward, ref_reward - model_reward)
                    all_reward_margins.append(reward_margin)

                    # chosen and rejected responses
                    chosen_response = torch.where(chosen_mask.unsqueeze(1), query_response, mixture_query_response)
                    rejected_response = torch.where(rejected_mask.unsqueeze(1), query_response, mixture_query_response)
                    all_chosen_responses.append(chosen_response)
                    all_rejected_responses.append(rejected_response)

                    ref_logprobs_model_response = self.compute_logprobs(
                        model, query_response, context_length, ref_model, mixture_coeff=args.mixture_coeff
                    )
                    ref_logprobs_ref_response = self.compute_logprobs(
                        model, mixture_query_response, context_length, ref_model, mixture_coeff=args.mixture_coeff
                    )

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
                gc.collect()

                # logging kl between the model and ref mixture model responses
                logprobs_model = logits_to_logprobs(
                    logits_responses, query_response[:, context_length:], args.temperature
                )
                logprobs_ref = logits_to_logprobs(
                    mixture_logits_responses, mixture_query_response[:, context_length:], args.temperature
                )
                kl = logprobs_model - logprobs_ref
                scores_margin = all_reward_margins.mean()

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

                            # total loss
                            loss = (dpo_losses).mean()

                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                chosen_rewards = self.beta * chosen_log_ratios.detach()
                                rejected_rewards = self.beta * rejected_log_ratios.detach()
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

            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = -logprobs_model.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                g_chosen_reward = self.accelerator.gather(chosen_rewards_stats)
                g_rejected_reward = self.accelerator.gather(rejected_rewards_stats)
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
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
                self.log(metrics)
            del (kl, mean_kl, mean_entropy, scores, scores_margin)

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

        logprobs = F.log_softmax(logits, dim=-1)
        target_ids = responses[:, context_length:]
        return torch.gather(logprobs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
