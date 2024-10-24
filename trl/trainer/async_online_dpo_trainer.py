# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import gc
import logging
import math
import os
import queue
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    is_vllm_available,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from .async_online_dpo_config import AsyncOnlineDPOConfig
from .utils import generate_model_card


if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams

    from ..vllm_utils import vllm_single_gpu_patch


INVALID_LOGPROB = 1.0
logger = logging.getLogger(__name__)


class AsyncOnlineDPOTrainer(Trainer):
    _tag_names = ["trl", "online-dpo", "async"]

    def __init__(
        self,
        config: AsyncOnlineDPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        if not is_vllm_available():
            raise ImportError("`vllm` library is required for AsyncOnlineDPOTrainer, please install vllm")

        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy
        self.beta = config.beta

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
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        # To be similar to online_dpo_trainer.py, our batch size refers to the number of prompts
        # This is unlike rloo_trainer.py where batch size is prompts * rloo_k (or in our case 2)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, reward_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = processing_class.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
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

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(self.processing_class),
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
            collate_fn=DataCollatorWithPadding(self.processing_class),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        loss_stats = torch.zeros(stats_shape, device=device)
        chosen_reward_stats = torch.zeros(stats_shape, device=device)
        chosen_logprobs_stats = torch.zeros(stats_shape, device=device)
        chosen_ref_logprobs_stats = torch.zeros(stats_shape, device=device)
        rejected_reward_stats = torch.zeros(stats_shape, device=device)
        rejected_logprobs_stats = torch.zeros(stats_shape, device=device)
        rejected_ref_logprobs_stats = torch.zeros(stats_shape, device=device)
        model.train()

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

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        if accelerator.is_main_process:
            if args.fp16:
                vllm_dtype = torch.float16
            elif args.bf16:
                vllm_dtype = torch.bfloat16
            else:
                vllm_dtype = torch.float32
            vllm_device = args.vllm_device or f"cuda:{accelerator.num_processes}"
            response_ids_Q = queue.Queue(maxsize=1)
            param_prompt_Q = queue.Queue(maxsize=1)
            thread = threading.Thread(
                target=vllm_generate,
                args=(
                    args.sft_model_path,
                    vllm_device,
                    args.vllm_gpu_memory_utilization,
                    vllm_dtype,
                    response_ids_Q,
                    param_prompt_Q,
                    args.temperature,
                    args.response_length,
                ),
            )
            thread.start()

        data = next(iter_dataloader)
        next_queries = data["input_ids"].to(device)
        next_queries = next_queries.repeat(2, 1)
        g_queries_list = gather_object(next_queries.tolist())
        if accelerator.is_main_process:
            g_queries_list = [
                [inneritem for inneritem in item if inneritem != processing_class.pad_token_id]
                for item in g_queries_list
            ]  # remove padding
            param_prompt_Q.put((None, g_queries_list))

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        for update in range(1, args.num_total_batches + 1):
            queries = next_queries
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            vllm_responses = torch.zeros(
                (args.batch_size * 2, args.response_length),
                device=accelerator.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                next_queries = data["input_ids"].to(device)
                next_queries = next_queries.repeat(2, 1)

                # with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                g_queries_list = gather_object(next_queries.tolist())
                if accelerator.is_main_process:
                    g_queries_list = [
                        [inneritem for inneritem in item if inneritem != processing_class.pad_token_id]
                        for item in g_queries_list
                    ]  # remove padding

                    # send next queries to be generated
                    model_named_parameters = accelerator._get_named_parameters(model)
                    param_prompt_Q.put((model_named_parameters.items(), g_queries_list))

                    # get response for previous queries
                    g_response_ids = response_ids_Q.get()

                    DUMMY_PAD_TOKEN = 0  # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
                    g_padded_response_ids = [
                        list(response) + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                        for response in g_response_ids
                    ]
                    g_padded_response_ids = torch.tensor(g_padded_response_ids, device=device)
                    vllm_responses[:] = g_padded_response_ids

                broadcast(vllm_responses, 0)
                local_vllm_responses = vllm_responses[
                    accelerator.local_process_index * queries.shape[0] : (accelerator.local_process_index + 1)
                    * queries.shape[0]
                ]

                context_length = queries.shape[1]
                query_responses = torch.cat((queries, local_vllm_responses), 1)
                responses = []
                postprocessed_responses = []
                scores = []
                sequence_lengths = []
                values = []
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del score
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)

                # 4. compute rewards
                num_examples = scores.size(0) // 2
                scores_reshaped = scores.reshape(2, num_examples).t()

                # Get the max scores and their local indices
                chosen_scores, chosen_local_indices = torch.max(scores_reshaped, dim=1)

                # Get the min scores and their local indices
                rejected_scores, rejected_local_indices = torch.min(scores_reshaped, dim=1)
                scores_margin = chosen_scores - rejected_scores

                # Calculate the global indices
                chosen_indices = chosen_local_indices * num_examples + torch.arange(num_examples, device=scores.device)
                rejected_indices = rejected_local_indices * num_examples + torch.arange(
                    num_examples, device=scores.device
                )
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
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
                        num_examples = chosen_mb_inds.shape[0]

                        # reference logprobs
                        with torch.no_grad():
                            concat_ref_output = forward(
                                ref_policy, concat_query_responses, processing_class.pad_token_id
                            )
                            chosen_ref_logits = concat_ref_output.logits[:num_examples]
                            rejected_ref_logits = concat_ref_output.logits[num_examples:]

                            chosen_ref_logits = chosen_ref_logits[:, context_length - 1 : -1]
                            chosen_ref_logits /= args.temperature + 1e-7
                            chosen_ref_all_logprobs = F.log_softmax(chosen_ref_logits, dim=-1)
                            chosen_ref_logprobs = torch.gather(
                                chosen_ref_all_logprobs, 2, chosen_responses.unsqueeze(-1)
                            ).squeeze(-1)
                            chosen_ref_logprobs = torch.masked_fill(
                                chosen_ref_logprobs, padding_mask[chosen_mb_inds], INVALID_LOGPROB
                            )
                            chosen_ref_logprobs_sum = (chosen_ref_logprobs * ~padding_mask[chosen_mb_inds]).sum(1)

                            rejected_ref_logits = rejected_ref_logits[:, context_length - 1 : -1]
                            rejected_ref_logits /= args.temperature + 1e-7
                            rejected_ref_all_logprobs = F.log_softmax(rejected_ref_logits, dim=-1)
                            rejected_ref_logprobs = torch.gather(
                                rejected_ref_all_logprobs, 2, rejected_responses.unsqueeze(-1)
                            ).squeeze(-1)
                            rejected_ref_logprobs = torch.masked_fill(
                                rejected_ref_logprobs, padding_mask[rejected_mb_inds], INVALID_LOGPROB
                            )
                            rejected_ref_logprobs_sum = (rejected_ref_logprobs * ~padding_mask[rejected_mb_inds]).sum(
                                1
                            )

                            ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

                        with accelerator.accumulate(model):
                            concat_output = forward(model, concat_query_responses, processing_class.pad_token_id)
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
                                chosen_logprobs, padding_mask[chosen_mb_inds], INVALID_LOGPROB
                            )
                            chosen_logprobs_sum = (chosen_logprobs * ~padding_mask[chosen_mb_inds]).sum(1)

                            # rejected
                            rejected_logits = rejected_logits[:, context_length - 1 : -1]
                            rejected_logits /= args.temperature + 1e-7
                            rejected_all_logprobs = F.log_softmax(rejected_logits, dim=-1)
                            rejected_logprobs = torch.gather(
                                rejected_all_logprobs, 2, rejected_responses.unsqueeze(-1)
                            ).squeeze(-1)
                            rejected_logprobs = torch.masked_fill(
                                rejected_logprobs, padding_mask[rejected_mb_inds], INVALID_LOGPROB
                            )
                            rejected_logprobs_sum = (rejected_logprobs * ~padding_mask[rejected_mb_inds]).sum(1)

                            pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum

                            logits = pi_logratios - ref_logratios

                            if self.args.loss_type == "sigmoid":
                                losses = -F.logsigmoid(self.beta * logits)
                            elif self.args.loss_type == "ipo":
                                losses = (logits - 1 / (2 * self.beta)) ** 2
                            else:
                                raise NotImplementedError(f"invalid loss type {self.args.loss_type}")

                            chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum).detach()
                            rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum).detach()

                            loss = losses.mean()
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss.detach()
                                chosen_reward_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    chosen_rewards.mean().detach()
                                )
                                chosen_logprobs_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    chosen_logprobs_sum.mean().detach()
                                )
                                chosen_ref_logprobs_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    chosen_ref_logprobs_sum.mean().detach()
                                )
                                rejected_reward_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    rejected_rewards.mean().detach()
                                )
                                rejected_logprobs_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    rejected_logprobs_sum.mean().detach()
                                )
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        logits, loss,
                        concat_output, concat_query_responses,
                        chosen_logits, rejected_logits,
                        chosen_logprobs, rejected_logprobs,
                        chosen_responses, rejected_responses,
                        chosen_all_logprobs, rejected_all_logprobs,
                        concat_ref_output,
                        chosen_ref_logits, rejected_ref_logits,
                        chosen_ref_logprobs, rejected_ref_logprobs,
                        chosen_ref_all_logprobs, rejected_ref_all_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(loss_stats).mean().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode

                # dpo metrics
                metrics["logps/chosen"] = self.accelerator.gather(chosen_logprobs_stats.mean()).mean().item()
                metrics["logps/rejected"] = self.accelerator.gather(rejected_logprobs_stats.mean()).mean().item()
                kl = (
                    (chosen_logprobs_stats - chosen_ref_logprobs_stats)
                    + (rejected_logprobs_stats - rejected_ref_logprobs_stats)
                ) / 2
                mean_kl = kl.mean()
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                mean_non_score_reward = (-self.beta * kl).mean()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                mean_rlhf_reward = scores.mean() + mean_non_score_reward
                metrics["objective/rlhf_reward"] = self.accelerator.gather(mean_rlhf_reward).mean().item()
                logprobs_sum = (chosen_logprobs_stats + rejected_logprobs_stats) / 2
                mean_entropy = -logprobs_sum.mean()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/scores_margin"] = self.accelerator.gather(scores_margin.mean()).mean().item()
                metrics["rewards/chosen"] = self.accelerator.gather(chosen_reward_stats.mean()).mean().item()
                metrics["rewards/rejected"] = self.accelerator.gather(rejected_reward_stats.mean()).mean().item()
                margin = chosen_reward_stats - rejected_reward_stats
                metrics["rewards/margins"] = self.accelerator.gather(margin.mean()).mean().item()
                accuracy = margin > 0
                metrics["rewards/accuracies"] = self.accelerator.gather(accuracy.float().mean()).mean().item()
                metrics["beta"] = self.beta

                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, mean_rlhf_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                values,
                sequence_lengths,
                contain_eos_token,
                response_idxs,
                padding_mask,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `List[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        # citation = textwrap.dedent("""\
        # @article{mziegler2019fine-tuning,
        #     title        = {{Fine-Tuning Language Models from Human Preferences}},
        #     author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
        #     year         = 2019,
        #     eprint       = {arXiv:1909.08593}
        # }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="AsyncOnlineDPO",
            # trainer_citation=citation,
            # paper_title="Fine-Tuning Language Models from Human Preferences",
            # paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))


def vllm_generate(
    model_name_or_path: str,
    vllm_device: str,
    vllm_gpu_memory_utilization: float,
    vllm_dtype: str,
    response_ids_Q: queue.Queue,
    param_prompt_Q: queue.Queue,
    temperature: float,
    response_length: int,
):
    vllm_single_gpu_patch()
    generation_config = SamplingParams(
        temperature=(temperature + 1e-7),
        top_p=1.0,
        max_tokens=response_length,
        include_stop_str_in_output=True,
    )

    llm = LLM(
        model=model_name_or_path,
        revision="main",
        tokenizer_revision="main",
        tensor_parallel_size=1,
        device=vllm_device,
        dtype=vllm_dtype,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    logger.info(f"🔥🔥🔥 vllm loaded in {vllm_dtype}")
    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
    i = 0
    while True:
        i += 1
        model_named_parameters, g_queries_list = param_prompt_Q.get()
        if model_named_parameters is None and g_queries_list is None:
            logger.info(
                "vllm thread received model params and queries = None, this indicates the end of training so exiting vllm thread"
            )
            break

        if i > 2:
            llmp.load_weights(model_named_parameters)

        outputs = llm.generate(prompt_token_ids=g_queries_list, sampling_params=generation_config, use_tqdm=False)
        response_token_ids = []
        for output in outputs:
            response_token_ids.append(output.outputs[0].token_ids)

        response_ids_Q.put(response_token_ids)
