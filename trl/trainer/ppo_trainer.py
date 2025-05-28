# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from transformers.utils import is_peft_available, is_rich_available
from transformers import AutoModelForSequenceClassification

from ..core import masked_mean, masked_whiten
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .ppo_config import PPOConfig
from .utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    empty_cache,
    exact_div,
    first_true_indices,
    forward,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb


INVALID_LOGPROB = 1.0


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits

@dataclass
class PPOStats:
    responses:float
    mean_kl:float
    objective_mean_entropy:float
    mean_non_score_reward: float
    rlhf_reward: float
    mean_scores: float
    policy_entropy_mean: Optional[torch.FloatTensor] = field(default=None)
    pg_loss: Optional[torch.FloatTensor] = field(default=None)
    vf_loss: Optional[torch.FloatTensor] = field(default=None)
    loss: Optional[torch.FloatTensor] = field(default=None)
    pg_clipfrac: Optional[torch.FloatTensor] = field(default=None)
    vf_clipfrac: Optional[torch.FloatTensor] = field(default=None)
    ratio_mean: Optional[torch.FloatTensor] = field(default=None)
    approxkl: Optional[torch.FloatTensor] = field(default=None)

@dataclass
class PPOComponents:
    responses: torch.IntTensor
    logprobs: torch.FloatTensor
    ref_logprobs: torch.FloatTensor
    values: torch.FloatTensor
    sequence_lengths: torch.FloatTensor
    scores: torch.FloatTensor
    rewards: torch.FloatTensor
    advantages: torch.FloatTensor
    returns: torch.FloatTensor
    padding_mask: torch.FloatTensor
    padding_mask_p1: torch.FloatTensor
    sequence_lengths_p1: torch.FloatTensor
    args: PPOConfig

    def process_ppo_components(self, postprocessed_responses: torch.IntTensor, eos_token_id: int) -> None:  
        """
        Processes the PPO components and initializes the necessary attributes.

        This method modifies the instance attributes `responses`, `logprobs`, `ref_logprobs`,
        `sequence_lengths`, `scores`, `values`, `padding_mask`, `sequence_lengths_p1`, and `padding_mask_p1`.
        """
        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.
        contain_eos_token = torch.any(postprocessed_responses == eos_token_id, dim=-1)
        if self.args.missing_eos_penalty is not None:
            self.scores[~contain_eos_token] -= self.args.missing_eos_penalty

        # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(self.responses.shape[1], device=self.responses.device).repeat(self.responses.shape[0], 1)
        self.padding_mask = response_idxs > self.sequence_lengths.unsqueeze(1)
        self.logprobs = torch.masked_fill(self.logprobs, self.padding_mask, INVALID_LOGPROB)
        self.ref_logprobs = torch.masked_fill(self.ref_logprobs, self.padding_mask, INVALID_LOGPROB)
        self.sequence_lengths_p1 = self.sequence_lengths + 1
        self.padding_mask_p1 = response_idxs > (self.sequence_lengths_p1.unsqueeze(1))
        self.values = torch.masked_fill(self.values, self.padding_mask_p1, 0)

    def compute_rewards(self) -> PPOStats:
        # 4. compute rewards
        # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
        logr = self.ref_logprobs - self.logprobs
        kl = -logr if self.args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
        non_score_reward = -self.args.kl_coef * kl
        self.rewards = non_score_reward.clone()
        actual_start = torch.arange(self.rewards.size(0), device=self.rewards.device)
        actual_end = torch.where(self.sequence_lengths_p1 < self.rewards.size(1), self.sequence_lengths_p1, self.sequence_lengths)
        self.rewards[[actual_start, actual_end]] += self.scores
        if self.args.whiten_rewards:
            self.rewards = masked_whiten(self.rewards, mask=~self.padding_mask_p1, shift_mean=False)
            self.rewards = torch.masked_fill(self.rewards, self.padding_mask_p1, 0)

        #calculate mean kl, mean entropy, mean non_score_reward, rlhf_reward and mean_scores for future stats logging
        mean_non_score_reward = non_score_reward.sum(1).mean()
        rlhf_reward = mean_non_score_reward + self.scores.mean()
        ppo_stats = PPOStats(
            mean_kl = kl.sum(1).mean(),
            objective_mean_entropy = (-self.logprobs).sum(1).mean(),
            mean_non_score_reward = mean_non_score_reward,
            rlhf_reward = rlhf_reward,
            mean_scores = self.scores.mean(),
            responses=self.responses,
        )
        return ppo_stats

    def compute_advantages_and_returns(self)-> None:
        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.responses.shape[1]
        for t in reversed(range(gen_length)):
            nextvalues = self.values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = self.rewards[:, t] + self.args.gamma * nextvalues - self.values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], axis=1)
        self.returns = advantages + self.values
        advantages = masked_whiten(advantages, ~self.padding_mask)
        self.advantages = torch.masked_fill(advantages, self.padding_mask, 0)
        empty_cache()
        gc.collect()
    
    def flush_unecessary_ppo_components_for_loss_computation(self)-> None:
        del self.scores, self.sequence_lengths,  self.ref_logprobs, self.rewards

@dataclass
class PPOLossComponents:
    micro_batch_inds: torch.FloatTensor
    vpred: torch.FloatTensor
    vpredclipped: torch.FloatTensor
    mb_return: torch.FloatTensor
    mb_logprobs: torch.FloatTensor
    mb_advantage: torch.FloatTensor
    logits: torch.FloatTensor
    new_logprobs: torch.FloatTensor

    def compute_loss(self, args: PPOConfig, padding_mask: torch.IntTensor, padding_mask_p1: torch.IntTensor, ppo_stats: PPOStats) -> PPOStats:
        vf_losses1 = torch.square(self.vpred - self.mb_return)
        vf_losses2 = torch.square(self.vpredclipped - self.mb_return)

        vf_loss_max = torch.max(vf_losses1, vf_losses2)
        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[self.micro_batch_inds])
        ppo_stats.vf_clipfrac = masked_mean(
            (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[self.micro_batch_inds]
        )
        del vf_losses1, vf_losses2, vf_loss_max 

        logprobs_diff = self.new_logprobs - self.mb_logprobs
        ratio = torch.exp(logprobs_diff)
        ppo_stats.approxkl = 0.5 * (logprobs_diff**2).mean()
        del logprobs_diff

        pg_losses = -self.mb_advantage * ratio
        pg_losses2 = -self.mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
        ppo_stats.ratio_mean = ratio.mean()

        del ratio
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = masked_mean(pg_loss_max, ~padding_mask[self.micro_batch_inds])

        ppo_stats.loss = pg_loss + args.vf_coef * vf_loss
        ppo_stats.pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padding_mask[self.micro_batch_inds])
        ppo_stats.pg_loss = pg_loss
        ppo_stats.vf_loss = vf_loss

        return ppo_stats

class PPOHandler:
    def __init__(
        self,
        responses: torch.IntTensor,
        postprocessed_responses: torch.IntTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        sequence_lengths: torch.IntTensor,
        scores: torch.FloatTensor,
        args: PPOConfig,
        eos_token_id: int,
    ):
        self.args = args
        self.eos_token_id = eos_token_id

        self.ppo_components = PPOComponents(
            responses=responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            sequence_lengths=sequence_lengths,
            scores=scores,
            args=args,
            rewards=None,
            advantages=None,
            returns=None,
            padding_mask=None,
            padding_mask_p1=None,
            sequence_lengths_p1=None
        )
        self.ppo_loss_components: PPOLossComponents = None
        self.ppo_stats: PPOStats = None

        self.ppo_components.process_ppo_components(postprocessed_responses, self.eos_token_id)
    
    def get_stats(self) -> PPOStats:
        return self.ppo_stats
    
    def compute_rewards(self):
        self.ppo_stats = self.ppo_components.compute_rewards()
    
    def compute_advantages_and_returns(self):
        self.ppo_components.compute_advantages_and_returns()

    def flush_unecessary_ppo_components_for_loss_computation(self):
        self.ppo_components.flush_unecessary_ppo_components_for_loss_computation()


    def compute_ppo_loss_components(self, model: PolicyAndValueWrapper, context_length: int, query_responses: torch.IntTensor, micro_batch_inds: int, pad_token_id: int)->None:
        mb_advantage = self.ppo_components.advantages[micro_batch_inds]
        mb_responses = self.ppo_components.responses[micro_batch_inds]
        mb_query_responses = query_responses[micro_batch_inds]
        mb_logprobs = self.ppo_components.logprobs[micro_batch_inds]
        mb_return = self.ppo_components.returns[micro_batch_inds]
        mb_values = self.ppo_components.values[micro_batch_inds]
        output, vpred_temp = forward(model, mb_query_responses, pad_token_id)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= self.args.temperature + 1e-7
        new_logprobs = selective_log_softmax(logits, mb_responses)
        new_logprobs = torch.masked_fill(
            new_logprobs, self.ppo_components.padding_mask[micro_batch_inds], INVALID_LOGPROB
        )
        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
        vpred = torch.masked_fill(vpred, self.ppo_components.padding_mask_p1[micro_batch_inds], 0)
        vpredclipped = torch.clamp(
                        vpred,
                        mb_values - self.args.cliprange_value,
                        mb_values + self.args.cliprange_value,
                    )
        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
        policy_entropy_mean = (torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)).mean()
        self.ppo_stats.policy_entropy_mean = policy_entropy_mean
        self.ppo_loss_components = PPOLossComponents(
            micro_batch_inds=micro_batch_inds,
            vpred=vpred,
            vpredclipped=vpredclipped,
            mb_return=mb_return,
            mb_logprobs=mb_logprobs,
            mb_advantage=mb_advantage,
            logits=logits,
            new_logprobs=new_logprobs
        )

    def compute_loss(self)->None:
        self.ppo_stats = self.ppo_loss_components.compute_loss(
            self.args,
            self.ppo_components.padding_mask,
            self.ppo_components.padding_mask_p1,
            self.ppo_stats
        )

    def get_loss(self)-> torch.FloatTensor:
        return self.ppo_stats.loss
    
    def flush_ppo_loss_components(self) ->None:
        del self.ppo_loss_components
    
    def flush_ppo_stats(self)->None:
        del self.ppo_stats


class PPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        args: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        ref_model: Optional[nn.Module],
        reward_model: Optional[nn.Module] = None,
        train_dataset: Optional[Dataset] = None,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
        use_only_step_method: bool = False,
        share_same_value_and_policy_lm_backbone: bool = False,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model
        self.use_only_step_method = use_only_step_method
        self.share_same_value_and_policy_lm_backbone = share_same_value_and_policy_lm_backbone

        if self.share_same_value_and_policy_lm_backbone:
            if value_model is not None:
                raise ValueError(
                    "You cannot pass a `value_model` when `share_same_value_and_policy_lm_backbone` is set to True."
                )
            value_model = AutoModelForSequenceClassification.from_config(
                self.policy_model.config
            )
            value_model.num_labels = 1
            value_model.classifier = nn.Linear(self.policy_model.config.hidden_size, 1)
            setattr(value_model, value_model.base_model_prefix, getattr(self.policy_model, self.policy_model.base_model_prefix))
        else:
            if value_model is None:
                raise ValueError(
                    "You must pass a `value_model` when `share_same_value_and_policy_lm_backbone` is set to False."
                )

        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        if self.args.stop_token and self.args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif self.args.stop_token:
            if self.args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {self.args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = self.args.stop_token_id

        if self.args.kl_estimator not in {"k1", "k3"}:
            raise ValueError(
                "kl_estimator must be either 'k1' (straightforward, unbiased) or 'k3' (lower variance, unbiased, "
                "appears to be a strictly better estimator). See "
                "[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) for details."
            )

        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if self.args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = self.args.model_adapter_name
        self.ref_adapter_name = self.args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.policy_model)

        self.reward_model = reward_model
        self.value_model = value_model
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None

        accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps)
        self.accelerator = accelerator
        self.args.world_size = accelerator.num_processes
        self.args.local_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        self.args.micro_batch_size = int(self.args.per_device_train_batch_size * self.args.world_size)
        self.args.batch_size = int(self.args.local_batch_size * self.args.world_size)
        self.args.mini_batch_size = exact_div(
            self.args.batch_size, self.args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        self.args.local_mini_batch_size = exact_div(
            self.args.local_batch_size, self.args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if self.args.whiten_rewards:
            assert self.args.local_mini_batch_size >= 8, (
                f"Per-rank minibatch size {self.args.local_mini_batch_size} is insufficient for whitening"
            )

        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()
        self.args.run_name = f"{self.args.exp_name}__{self.args.seed}__{time_int}"
        self.local_seed = self.args.seed + accelerator.process_index * 100003

        if self.args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, self.args.num_total_batches // self.args.num_sample_generations)
        self.local_dataloader_batch_size = self.args.local_batch_size

        for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
            if module is not None:
                disable_dropout_in_model(module)

        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model.config = self.policy_model.config

        if not self.use_only_step_method:
            self.default_init_dataloader(train_dataset, data_collator, eval_dataset)
        else:
            warnings.warn(
                "You are using the PPOTrainer with `use_only_step_method=True`. "
                "Be sure to handle the dataloader manually",
            )
            self.args.num_total_batches = math.ceil(
                self.args.total_episodes / self.args.batch_size
            )  # we may train for more than `total_episodes`

        self.create_optimizer_and_scheduler(num_training_steps=self.args.num_total_batches)
        self.default_callbacks_init(callbacks)

        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        torch.manual_seed(self.args.seed)
        if not self.use_only_step_method:
            self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        else:
            self.model, self.optimizer = accelerator.prepare(self.model, self.optimizer)
        torch.manual_seed(self.local_seed)

        self.prepare_training_gpu()

    def default_init_dataloader(self, train_dataset, data_collator, eval_dataset) -> None:
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        if self.args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            self.args.total_episodes = int(self.args.num_train_epochs * self.train_dataset_len)
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
    
    def default_callbacks_init(self, callbacks) -> None:
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

    def prepare_training_gpu(self) -> None:
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, self.args.per_device_train_batch_size, self.args.fp16, self.args.bf16
                )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, self.args.per_device_train_batch_size, self.args.fp16, self.args.bf16
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model.policy).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def initialize_training(self) -> None:
        """
        Initialize the training for trainer or manual stepping
        """
        args = self.args

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

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
        
        self.accelerator.print(f"===training policy===")
        self.start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        device = self.accelerator.device
        self.approxkl_stats = torch.zeros(stats_shape, device=device)
        self.pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.pg_loss_stats = torch.zeros(stats_shape, device=device)
        self.vf_loss_stats = torch.zeros(stats_shape, device=device)
        self.vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.entropy_stats = torch.zeros(stats_shape, device=device)
        self.ratio_stats = torch.zeros(stats_shape, device=device)
        self.model.train()



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
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
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
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    def get_logprobs_per_rollout(
            self, 
            queries: torch.IntTensor, 
            query_responses: torch.IntTensor, 
            logitss: torch.FloatTensor, 
            index: int
        ) -> Tuple[torch.IntTensor, torch.IntTensor, torch.IntTensor, torch.FloatTensor, torch.FloatTensor]:
        query = queries[index : index + self.args.local_rollout_forward_batch_size]
        query_response = query_responses[index : index + self.args.local_rollout_forward_batch_size]
        response = query_response[:, self.context_length:]
        logits = logitss[index : index + self.args.local_rollout_forward_batch_size]
        logprob = selective_log_softmax(logits, response)
        del logits
        empty_cache()

        if self.ref_model is None:
            with self.null_ref_context():
                ref_output = forward(self.model.policy, query_response, self.processing_class.pad_token_id)
        else:
            ref_output = forward(self.ref_model, query_response, self.processing_class.pad_token_id)
        ref_logits = ref_output.logits[:, self.context_length - 1 : -1]
        ref_logits /= self.args.temperature + 1e-7
        ref_logprob = selective_log_softmax(ref_logits, response)
        del ref_output, ref_logits
        empty_cache()
        return query, query_response, response, logprob, ref_logprob

    def get_ppo_handler(
            self, 
            queries: torch.IntTensor, 
            query_responses: torch.IntTensor, 
            logitss: torch.FloatTensor, 
            scores: Optional[torch.FloatTensor] = None
        ) -> PPOHandler:

        responses = []
        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        sequence_lengths = []
        values = []
        if scores is None:
            scores = []

        for i in range(0, queries.shape[0], self.args.local_rollout_forward_batch_size):
            # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
            query, query_response, response, logprob, ref_logprob = self.get_logprobs_per_rollout(
                queries, query_responses, logitss, i
            )
            postprocessed_response = response
            if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                postprocessed_response = truncate_response(
                    self.stop_token_id, self.processing_class.pad_token_id, response
                )
            postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
            sequence_length = first_true_indices(postprocessed_response == self.processing_class.pad_token_id) - 1
            unwrapped_value_model = self.accelerator.unwrap_model(self.model).value_model
            full_value, _, _ = get_reward(
                unwrapped_value_model, query_response, self.processing_class.pad_token_id, self.context_length
            )
            value = full_value[:, self.context_length - 1 : -1].squeeze(-1)

            if self.reward_model:
                # Response Processing 2. run reward model on the truncated responses
                _, score, _ = get_reward(
                    self.reward_model, postprocessed_query_response, self.processing_class.pad_token_id, self.context_length
                )
                scores.append(score)
            responses.append(response)
            postprocessed_responses.append(postprocessed_response)
            logprobs.append(logprob)
            ref_logprobs.append(ref_logprob)
            sequence_lengths.append(sequence_length)
            values.append(value)

        responses = torch.cat(responses, 0)
        postprocessed_responses = torch.cat(postprocessed_responses, 0)
        logprobs = torch.cat(logprobs, 0)
        ref_logprobs = torch.cat(ref_logprobs, 0)
        sequence_lengths = torch.cat(sequence_lengths, 0)
        values = torch.cat(values, 0)
        if isinstance(scores, list):
            scores = torch.cat(scores, 0)
        del (logprob, ref_logprob, full_value, value, score, unwrapped_value_model)
        empty_cache()
        gc.collect()

        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.

        return PPOHandler(
            responses=responses,
            postprocessed_responses=postprocessed_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            sequence_lengths=sequence_lengths,
            scores=scores,
            args=self.args,
            eos_token_id=self.processing_class.eos_token_id)
    
    def save_training_stats(self, ppo_handler: PPOHandler, ppo_epoch_idx: int, minibatch_idx: int, gradient_accumulation_idx: int) -> None:
        with torch.no_grad():
            self.approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ppo_handler.ppo_stats.approxkl
            self.pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                ppo_handler.ppo_stats.pg_clipfrac
            )
            self.pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ppo_handler.ppo_stats.pg_loss
            self.vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ppo_handler.ppo_stats.vf_loss
            self.vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                ppo_handler.ppo_stats.vf_clipfrac
            )
            self.entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ppo_handler.ppo_stats.policy_entropy_mean
            self.ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ppo_handler.ppo_stats.ratio_mean               
    
    def update_training_model(self, query_responses: torch.IntTensor, ppo_handler: PPOHandler) -> None:
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(self.args.num_ppo_epochs):
            b_inds = np.random.permutation(self.args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, self.args.local_batch_size, self.args.local_mini_batch_size):
                mini_batch_end = mini_batch_start + self.args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, self.args.local_mini_batch_size, self.args.per_device_train_batch_size):
                    with self.accelerator.accumulate(self.model):
                        micro_batch_end = micro_batch_start + self.args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                        ppo_handler.compute_ppo_loss_components(
                            model=self.model,
                            context_length=self.context_length,
                            query_responses=query_responses, 
                            micro_batch_inds=micro_batch_inds,
                            pad_token_id=self.processing_class.pad_token_id
                        )

                        # calculate the loss
                        ppo_handler.compute_loss()

                        self.save_training_stats(ppo_handler, ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx)

                        self.accelerator.backward(ppo_handler.get_loss())
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    gradient_accumulation_idx += 1
                # fmt: off
                ppo_handler.flush_ppo_loss_components()
                # fmt: on
                empty_cache()
                gc.collect()

                minibatch_idx += 1


    def log_stats(self, ppo_handler: PPOHandler) -> None:
        with torch.no_grad():
            eps = int(self.state.episode / (time.time() - self.start_time))
            metrics = {}
            metrics["eps"] = eps
            metrics["objective/kl"] = self.accelerator.gather_for_metrics(ppo_handler.ppo_stats.mean_kl).mean().item()
            metrics["objective/entropy"] = self.accelerator.gather_for_metrics(ppo_handler.ppo_stats.objective_mean_entropy).mean().item()
            metrics["objective/non_score_reward"] = (
                self.accelerator.gather_for_metrics(ppo_handler.ppo_stats.mean_non_score_reward).mean().item()
            )
            metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(ppo_handler.ppo_stats.rlhf_reward).mean().item()
            metrics["objective/scores"] = self.accelerator.gather_for_metrics(ppo_handler.ppo_stats.mean_scores).mean().item()
            metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(self.approxkl_stats).mean().item()
            metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(self.pg_clipfrac_stats).mean().item()
            metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(self.pg_loss_stats).mean().item()
            metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(self.vf_loss_stats).mean().item()
            metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(self.vf_clipfrac_stats).mean().item()
            metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(self.entropy_stats).mean().item()
            metrics["val/ratio"] = self.accelerator.gather_for_metrics(self.ratio_stats).mean().item()
            metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(self.ratio_stats).var().item()
            metrics["val/num_eos_tokens"] = (ppo_handler.ppo_stats.responses == self.processing_class.eos_token_id).sum().item()
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["episode"] = self.state.episode
            self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
            self.state.global_step += 1
            self.log(metrics)

    def input_step_shape_check(
            self, 
            queries: torch.IntTensor, 
            query_responses: torch.IntTensor,  
            scores: Optional[torch.FloatTensor] = None, 
            query_responses_logitss: Optional[torch.FloatTensor] = None,
    ) -> None:
        # Check exclusivity of scores and reward_model
        if scores is None and self.reward_model is None:
            raise ValueError(
                "Either `scores` or `reward_model` must be provided."
            )
        if scores is not None and self.reward_model is not None:
            raise ValueError(
                "Both `scores` and `reward_model` are provided. Please provide only one."
            )

        # Check dimensionality
        for tensor, name, expected_dim in zip(
            [query_responses, queries, scores, query_responses_logitss],
            ["query_responses", "queries", "scores", "query_responses_logitss"],
            [2, 2, 2, 3]
        ):
            if tensor is not None and tensor.dim() != expected_dim:
                raise ValueError(f"{name} should be a {expected_dim}D tensor, but got {tensor.dim()}D.")

        # Check batch consistency
        batch_size = queries.shape[0]
        if query_responses.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: `queries` has {batch_size}, `query_responses` has {query_responses.shape[0]}.")
        if scores is not None and scores.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: `scores` has {scores.shape[0]}, expected {batch_size}.")

        # Check sequence alignment
        query_part = query_responses[:, :self.context_length]
        if query_part.shape[1] != queries.shape[1]:
            raise ValueError(
                f"Mismatch between query portion of `query_responses` (shape: {query_part.shape[1]}) and `queries` (shape: {queries.shape[1]})."
            )

        if scores is not None:
            if scores.shape[1] != query_part.shape[1]:
                raise ValueError(
                    f"Mismatch between `scores` (shape: {scores.shape[1]}) and query portion of `query_responses` (shape: {query_part.shape[1]})."
                )
            else:
                warnings.warn(
                    "Shapes of `query_responses` and `scores` match, but padding or masking inconsistencies may still lead to incorrect results. Please verify your preprocessing."
                )

    def step(
            self, 
            queries: torch.IntTensor, 
            query_responses: torch.IntTensor,  
            scores: Optional[torch.FloatTensor] = None, 
            query_responses_logitss: Optional[torch.FloatTensor] = None,
            check_input_shape: Optional[bool] = True,
        ) -> None:
        
        if not hasattr(self, "approxkl_stats"):
            self.initialize_training()
        device = self.accelerator.device
        self.state.episode += 1 * self.args.batch_size
        self.context_length = queries.shape[1]

        if check_input_shape:
            self.input_step_shape_check(queries, query_responses, scores, query_responses_logitss)

        with torch.no_grad():
            queries.to(device)
            query_responses.to(device)

            if query_responses_logitss is None:
                output = forward(self.model, query_responses, self.processing_class.pad_token_id)
                logitss = output.logits
            else:
                logitss = query_responses_logitss

            ppo_handler = (
                self.get_ppo_handler(queries, query_responses, logitss, scores)
            )
            del logitss
            ppo_handler.compute_rewards()
            ppo_handler.compute_advantages_and_returns()
            del scores
            ppo_handler.flush_unecessary_ppo_components_for_loss_computation()
            empty_cache()
        
        self.update_training_model(query_responses, ppo_handler)
        self.log_stats(ppo_handler)
        self.lr_scheduler.step()
        ppo_handler.flush_ppo_stats()
        

    def train(self):
        if self.reward_model is None:
            raise ValueError("`reward_model` must be provided for training.")
        if self.train_dataset is None:
            raise ValueError("`train_dataset` must be provided for training.")

        device = self.accelerator.device

        def repeat_generator():
            while True:
                yield from self.dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(self.args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        self.initialize_training()

        for update in range(1, self.args.num_total_batches + 1):
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        self.args.local_rollout_forward_batch_size,
                        self.processing_class.pad_token_id,
                        generation_config,
                    )

            self.step(queries, query_responses, query_responses_logitss=logitss)
            empty_cache()
            gc.collect()

            self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(self.model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            if self.args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(self.model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
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

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
