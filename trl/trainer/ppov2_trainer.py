import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback


from . import PolicyTrainerBase, PolicyTrainerArguments


INVALID_LOGPROB = 1.0


"""
python -i trl/trainer/ppov2_trainer.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type linear \
"""


@dataclass
class PPOV2Config(PolicyTrainerArguments):
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange: float = 0.2
    """the clip range"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
       output = self.critic_backbone(
            **kwargs,
        )
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


# PR TODO: PPOV2 trainer init subclassing PolicyTrainerBase
class PPOV2Trainer(Trainer):
    def __init__(
        self,
        args: PPOV2Config,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        train_generation_config: GenerationConfig,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        eval_generation_config: Optional[GenerationConfig] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # model_init: Optional[Callable[[torch.nn.Module], None]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.policy = policy

        self.policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding


        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.train_generation_config = train_generation_config
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.eval_generation_config = eval_generation_config
        if eval_generation_config is None:
            self.eval_generation_config = train_generation_config
        self.optimizer, self.lr_scheduler = optimizers
        self.callbacks = callbacks

        #########
        # calculate various batch sizes
        #########
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.nminibatches
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(args.batch_size, args.nminibatches)
        args.local_mini_batch_size = exact_div(args.local_batch_size, args.nminibatches)
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_updates = args.total_episodes // args.batch_size
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime

        #########
        # disable dropout
        #########
        for module in [policy, ref_policy, value_model, reward_model]:
            disable_dropout(module)
        policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        #########
        # setup model, optimizer, and others
        #########
        if args.truncate_token and args.truncate_token == "eos":
            args.truncate_token_id = tokenizer.eos_token_id
        self.model = PolicyAndValueWrapper(policy, value_model)
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        DEFAULT_CALLBACKS = [DefaultFlowCallback]
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        if self.callbacks is None:
            self.callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

    # def get_train_dataloader(self) -> DataLoader:

    def save_model(self, output_dir: Optional[str] = None):
        """
        Copied from Trainer.save_model, simplified.
        By default we only save the policy and not the value network.
        """
        unwrapped: PreTrainedModel = self.accelerator.unwrap_model(self.model).policy
        if output_dir is None:
            output_dir = self.args.output_dir
        state_dict = self.accelerator.get_state_dict(unwrapped)
        if self.args.should_save:
            self._save(output_dir, state_dict=state_dict)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/trainer.py#L3112
        """
        model.train()

        inputs = self._prepare_inputs(inputs)
        queries = inputs["input_ids"].to(self.accelerator.device)
        queries = queries.repeat(self.args.rloo_k, 1)

        context_length = queries.shape[1]
        query_responses, logits = self.generate(
            self.model,
            queries,
            self.train_generation_config,
        )
        responses = torch.stack([query_response[context_length:] for query_response in query_responses], dim=0)

        all_logprobs = F.log_softmax(logits, dim=-1)
        logprobs = torch.gather(all_logprobs, -1, responses.unsqueeze(-1)).squeeze(-1)
        del logits, all_logprobs

        with torch.no_grad():
            with self.ref_model_mgr as ref_model:
                ref_output_logits = self.forward(ref_model, query_responses).logits
        ref_logits = ref_output_logits[:, context_length - 1 : -1]
        ref_logits /= self.args.temperature + 1e-7
        ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
        ref_logprobs = torch.gather(ref_all_logprobs, -1, responses.unsqueeze(-1)).squeeze(-1)
        del ref_output_logits, ref_logits, ref_all_logprobs

        # Response Processing 1. truncate response after the
        # first occurrence of `truncate_token_id`
        postprocessed_responses = responses
        if self.args.truncate_token_id:
            postprocessed_responses = self.truncate_response(responses)

        # Response Processing 2. run reward model on the truncated responses
        postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
        sequence_lengths = first_true_indices(postprocessed_responses == self.tokenizer.pad_token_id) - 1

        # PR TODO: everything preceding this is common with RLOOTrainer.training_step, apply DRY

        full_value, _, _ = self.get_reward(
            accelerator.unwrap_model(model).value_model, query_responses, context_length
        )
        values = full_value[:, context_length - 1 : -1].squeeze(-1)
        _, score, _ = self.get_reward(self.reward_model, postprocessed_query_response, tokenizer, context_length)
        torch.cuda.empty_cache()

        # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
        # responses not passing that filter will receive a low (fixed) score
        # only query humans on responses that pass that filter
        contain_eos_token = torch.any(postprocessed_responses == self.tokenizer.eos_token_id, dim=-1)
        if self.args.non_eos_penalty:
            scores = torch.where(contain_eos_token, scores, torch.full_like(scores, self.args.penalty_reward_value))
        # PR TODO: this is from original, but maybe it should be logged somewhere?
        #self.accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

        # be very careful with `padding_mask`;
        # see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        sequence_lengths_p1 = sequence_lengths + 1
        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
        values = torch.masked_fill(values, padding_mask_p1, 0)

        # 4. compute rewards
        kl = logprobs - ref_logprobs
        non_score_reward = (-self.args.kl_coef * kl).sum(1)
        rewards = non_score_reward.clone()
        actual_start = torch.arange(rewards.size(0), device=rewards.device)
        actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)

        # 5. whiten rewards
        if self.args.whiten_rewards:
            rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
            rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

        # 6. compute advantages and returns
        lastgaelam = 0
        advantages_reversed = []
        gen_length = responses.shape[1]
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], axis=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, ~padding_mask)
        advantages = torch.masked_fill(advantages, padding_mask, 0)
        torch.cuda.empty_cache()

        # calculate loss
        output, vpred_temp = self.forward(model, query_responses)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= self.args.temperature + 1e-7
        new_all_logprobs = F.log_softmax(logits, dim=-1)
        new_logprobs = torch.gather(new_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
        new_logprobs = torch.masked_fill(
            new_logprobs, padding_mask, INVALID_LOGPROB
        )

        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
        vpred = torch.masked_fill(vpred, padding_mask_p1, 0)
        vpredclipped = torch.clamp(
            vpred,
            mb_values - self.args.cliprange_value,
            mb_values + self.args.cliprange_value,
        )

        vf_losses1 = torch.square(vpred - mb_return)
        vf_losses2 = torch.square(vpredclipped - mb_return)
        vf_loss_max = torch.max(vf_losses1, vf_losses2)
        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1)
        vf_clipfrac = masked_mean(
            (vf_losses2 > vf_losses1).float(), ~padding_mask_p1
        )
        logprobs_diff = new_logprobs - mb_logprobs
        ratio = torch.exp(logprobs_diff)
        pg_losses = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(
            ratio,
            1.0 - self.args.cliprange, 1.0 + self.args.cliprange
        )
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = masked_mean(pg_loss_max, ~padding_mask)
        pg_clipfrac = masked_mean(
            (pg_losses2 > pg_losses).float(),
            ~padding_mask
        )
        loss = pg_loss + args.vf_coef * vf_loss

        accelerator.backward(loss)

        # calculate stats
        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
        approxkl = 0.5 * (logprobs_diff**2).mean()
        mean_kl = kl.sum(1).mean()
        mean_entropy = (-logprobs).sum(1).mean()
        mean_non_score_reward = non_score_reward.sum(1).mean()

        self.log({
            "objective/kl": self.accelerator.gather(mean_kl).mean().item(),
            "objective/entropy": self.accelerator.gather(mean_entropy).mean().item(),
            "objective/rlhf_reward": self.accelerator.gather(rlhf_reward).mean().item(),
            "objective/scores": self.accelerator.gather(scores.mean()).mean().item(),
            "policy/approxkl_avg": self.accelerator.gather(approxkl).mean().item(),
            "policy/clipfrac_avg": self.accelerator.gather(pg_clipfrac).mean().item(),
            "loss/policy_avg": self.accelerator.gather(pg_loss).mean().item(),
            # PR TODO: this isn't calculated in the original
            #"loss/value_avg": self.accelerator.gather(vf_loss_stats).mean().item(),
            #"val/clipfrac_avg": self.accelerator.gather(vf_clipfrac_stats).mean().item(),

            # PR TODO: how does this differ from mean_entropy
            #"policy/entropy_avg": self.accelerator.gather(entropy).mean().item(),
            "val/ratio": self.accelerator.gather(new_ratio).mean().item(),

            # PR TODO
            #"val/ratio_var": self.accelerator.gather(ratio_stats).var().item(),
            "val/num_eos_tokens": (responses == self.tokenizer.eos_token_id).sum().item(),
        })

        ret_loss = loss.detach()

        # PR TOOD: del vars
        torch.cuda.empty_cache()

        return ret_loss
