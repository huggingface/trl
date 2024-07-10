import gc
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.rloo_trainer import INVALID_LOGPROB, RLOOTrainer
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)

from .online_dpo_config import OnlineDPOConfig


@dataclass
class OnlineTrainerState(TrainerState):
    episode: int = 0


class OnlineDPOTrainer(RLOOTrainer):
    def __init__(
        self,
        config: OnlineDPOConfig,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        loss_type: str = "sigmoid",
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # model_init: Optional[Callable[[torch.nn.Module], None]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
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
            args.total_episodes = args.num_train_epochs * self.train_dataset_len
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size,
            args.num_mini_batches,
            "`batch_size` must be a multiple of `num_mini_batches`",
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`",
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_updates = args.total_episodes // args.batch_size
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)

        assert args.rloo_k == 2, "currently only supports generating 2 completions per prompt"
        self.local_dataloader_batch_size = args.local_batch_size

        ### DPO stuff
        self.beta = config.beta
        self.loss_type = config.loss_type

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, reward_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks,
            self.model,
            self.tokenizer,
            self.optimizer,
            self.lr_scheduler,
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
                self.reward_model,
                args.per_device_train_batch_size,
                config.fp16,
                config.bf16,
            )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy,
                args.per_device_train_batch_size,
                config.fp16,
                config.bf16,
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

        self.ref_model = self.ref_policy

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
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

        accelerator.print("===training policy===")
        self.state.global_step = 0
        self.state.episode = 0
        start_time = time.time()
        stats_shape = (
            args.num_ppo_epochs,
            args.num_mini_batches,
            args.gradient_accumulation_steps,
        )
        loss_stats = torch.zeros(stats_shape, device=device)

        model.train()
        self.state.max_steps = args.num_updates * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

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
        saved_data = {"prompt": [], "chosen": [], "rejected": [], "update": []}

        for update in range(1, args.num_updates + 1):
            self.state.episode += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                queries = queries.repeat(args.rloo_k, 1)
                context_length = queries.shape[1]
                query_responses = []
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                        query = queries[i : i + args.local_rollout_forward_batch_size]
                        query_response, logits = generate(
                            unwrapped_model,
                            query,
                            tokenizer.pad_token_id,
                            generation_config,
                        )
                        response = query_response[:, context_length:]
                        all_logprob = F.log_softmax(logits, dim=-1)
                        logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        del logits, all_logprob
                        torch.cuda.empty_cache()

                        ref_output = forward(ref_policy, query_response, tokenizer.pad_token_id)
                        ref_logits = ref_output.logits[:, context_length - 1 : -1]
                        ref_logits /= args.temperature + 1e-7
                        ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                        ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                        del ref_output, ref_logits, ref_all_logprob
                        torch.cuda.empty_cache()

                        # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                        postprocessed_response = response
                        if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                            postprocessed_response = truncate_response(
                                args.stop_token_id, tokenizer.pad_token_id, response
                            )

                        # Response Processing 2. run reward model on the truncated responses
                        postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                        sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                        _, score, _ = get_reward(
                            reward_model,
                            postprocessed_query_response,
                            tokenizer.pad_token_id,
                            context_length,
                        )

                        query_responses.append(query_response)
                        responses.append(response)
                        postprocessed_responses.append(postprocessed_response)
                        logprobs.append(logprob)
                        ref_logprobs.append(ref_logprob)
                        sequence_lengths.append(sequence_length)
                        scores.append(score)
                query_responses = torch.cat(query_responses, 0)
                responses = torch.cat(responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del (logprob, ref_logprob, score)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
                if args.non_eos_penalty:
                    scores = torch.where(
                        contain_eos_token,
                        scores,
                        torch.full_like(scores, args.penalty_reward_value),
                    )
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                kl = logprobs - ref_logprobs
                non_score_reward = (-args.kl_coef * kl).sum(1)
                rlhf_reward = scores + non_score_reward

                # num_examples should be same as args.local_batch_size
                num_examples = scores.size(0) // 2
                first_half = scores[:num_examples]
                second_half = scores[num_examples:]

                num_examples_range = torch.arange(num_examples).to(scores.device)

                chosen_indices = torch.where(
                    first_half >= second_half,
                    num_examples_range.clone(),
                    num_examples_range.clone() + num_examples,
                )
                rejected_indices = torch.where(
                    first_half < second_half,
                    num_examples_range.clone(),
                    num_examples_range.clone() + num_examples,
                )

                scores_margin = scores[chosen_indices] - scores[rejected_indices]

                if self.args.save_generations:
                    decoded_queries = tokenizer.batch_decode(queries[:num_examples], skip_special_tokens=True)
                    decoded_chosen = tokenizer.batch_decode(postprocessed_responses[chosen_indices])
                    decoded_rejected = tokenizer.batch_decode(postprocessed_responses[rejected_indices])

                    # WARNING, if pad token == eos token, this will remove the eos from the end
                    decoded_chosen = [r.split(tokenizer.pad_token)[0] for r in decoded_chosen]
                    decoded_rejected = [r.split(tokenizer.pad_token)[0] for r in decoded_rejected]

                    saved_data["prompt"].extend(gather_object(decoded_queries))
                    saved_data["chosen"].extend(gather_object(decoded_chosen))
                    saved_data["rejected"].extend(gather_object(decoded_rejected))
                    saved_data["update"].extend(gather_object([update for _ in range(num_examples)]))

                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                all_chosen_rewards = []
                all_rejected_rewards = []
                all_chosen_logprobs = []
                all_rejected_logprobs = []
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            ## chosen
                            chosen_mb_inds = chosen_indices[micro_batch_inds]
                            chosen_responses = responses[chosen_mb_inds]

                            ## rejected
                            rejected_mb_inds = rejected_indices[micro_batch_inds]
                            rejected_responses = responses[rejected_mb_inds]

                            concat_mb_inds = torch.cat((chosen_mb_inds, rejected_mb_inds), dim=0)
                            concat_query_responses = query_responses[concat_mb_inds]

                            concat_output = forward(model, concat_query_responses, tokenizer.pad_token_id)
                            num_examples = chosen_mb_inds.shape[0]
                            chosen_logits = concat_output.logits[:num_examples]
                            rejected_logits = concat_output.logits[num_examples:]

                            # chosen
                            chosen_logits = chosen_logits[:, context_length - 1 : -1]
                            chosen_logits /= args.temperature + 1e-7
                            chosen_all_logprobs = F.log_softmax(chosen_logits, dim=-1)
                            chosen_logprobs = torch.gather(
                                chosen_all_logprobs, 2, chosen_responses.unsqueeze(-1)
                            ).squeeze(-1)
                            chosen_logprobs = torch.masked_fill(
                                chosen_logprobs,
                                padding_mask[chosen_mb_inds],
                                INVALID_LOGPROB,
                            )
                            chosen_ref_logprobs = ref_logprobs[chosen_mb_inds]
                            chosen_logprobs_sum = (chosen_logprobs * ~padding_mask[chosen_mb_inds]).sum(1)
                            chosen_ref_logprobs_sum = (chosen_ref_logprobs * ~padding_mask[chosen_mb_inds]).sum(1)

                            # rejected
                            rejected_logits = rejected_logits[:, context_length - 1 : -1]
                            rejected_logits /= args.temperature + 1e-7
                            rejected_all_logprobs = F.log_softmax(rejected_logits, dim=-1)
                            rejected_logprobs = torch.gather(
                                rejected_all_logprobs,
                                2,
                                rejected_responses.unsqueeze(-1),
                            ).squeeze(-1)
                            rejected_logprobs = torch.masked_fill(
                                rejected_logprobs,
                                padding_mask[rejected_mb_inds],
                                INVALID_LOGPROB,
                            )
                            rejected_ref_logprobs = ref_logprobs[rejected_mb_inds]
                            rejected_logprobs_sum = (rejected_logprobs * ~padding_mask[rejected_mb_inds]).sum(1)
                            rejected_ref_logprobs_sum = (rejected_ref_logprobs * ~padding_mask[rejected_mb_inds]).sum(
                                1
                            )

                            pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
                            ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

                            logits = pi_logratios - ref_logratios

                            if self.loss_type == "sigmoid":
                                losses = -F.logsigmoid(self.beta * logits)
                            elif self.loss_type == "ipo":
                                losses = (logits - 1 / (2 * self.beta)) ** 2
                            else:
                                raise NotImplementedError(f"invalid loss type {self.loss_type}")

                            chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum).detach()
                            rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum).detach()

                            loss = losses.mean()
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss.detach()
                            all_chosen_rewards.append(chosen_rewards)
                            all_chosen_logprobs.append(chosen_logprobs_sum)
                            all_rejected_rewards.append(rejected_rewards)
                            all_rejected_logprobs.append(rejected_logprobs_sum)
                            #     entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            #     ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    self.state.global_step += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        loss, logits,
                        concat_output, concat_query_responses,
                        chosen_logits, rejected_logits,
                        chosen_logprobs, rejected_logprobs,
                        chosen_responses, rejected_responses,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()

            all_chosen_rewards = torch.cat(all_chosen_rewards, 0)
            all_rejected_rewards = torch.cat(all_rejected_rewards, 0)
            all_chosen_logprobs = torch.cat(all_chosen_logprobs, 0)
            all_rejected_logprobs = torch.cat(all_rejected_logprobs, 0)

            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                # policy_chosen_logps = logprobs[chosen_indices]
                # policy_rejected_logps = logprobs[rejected_indices]

                chosen_rewards = self.accelerator.gather(all_chosen_rewards)
                chosen_logprobs = self.accelerator.gather(all_chosen_logprobs)
                rejected_rewards = self.accelerator.gather(all_rejected_rewards)
                rejected_logprobs = self.accelerator.gather(all_rejected_logprobs)

                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["objective/scores_margin"] = self.accelerator.gather(scores_margin.mean()).mean().item()
                metrics["rewards/chosen"] = chosen_rewards.mean().item()
                metrics["rewards/rejected"] = rejected_rewards.mean().item()
                metrics["rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
                metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
                metrics["logps/rejected"] = rejected_logprobs.mean().item()
                metrics["logps/chosen"] = chosen_logprobs.mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(loss_stats).mean().item()
                # metrics["logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
                # metrics["logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
                metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.log(metrics)
            del (
                kl,
                mean_kl,
                mean_entropy,
                scores,
                scores_margin,
                all_chosen_rewards,
                all_chosen_logprobs,
                all_rejected_rewards,
                all_rejected_logprobs,
            )

            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        if self.args.save_generations:
            if accelerator.is_local_main_process:
                dataset = Dataset.from_dict(saved_data)
                dataset.save_to_disk(os.path.join(self.args.output_dir, "online_dataset"))

    def generate_completions(self, sampling: bool = False):
        args = self.args
        tokenizer = self.tokenizer
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        for batch in self.eval_dataloader:
            query = batch["input_ids"]
            with torch.no_grad():
                context_length = query.shape[1]
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    query_response, _ = generate(
                        unwrapped_model,
                        query,
                        tokenizer.pad_token_id,
                        generation_config,
                    )
                response = query_response[:, context_length:]
                postprocessed_response = response
                if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(args.stop_token_id, tokenizer.pad_token_id, response)
                table["query"].extend(gather_object(tokenizer.batch_decode(query, skip_special_tokens=True)))
                table["model response"].extend(gather_object(tokenizer.batch_decode(postprocessed_response)))

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                _, score, _ = get_reward(
                    self.reward_model,
                    postprocessed_query_response,
                    tokenizer.pad_token_id,
                    context_length,
                )
                table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df.iloc[0 : 0 + 5])
        if "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})
