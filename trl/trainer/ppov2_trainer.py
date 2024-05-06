from accelerate.utils import gather_object
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationConfig
from trl.models.utils import unwrap_model_for_generation

from . import PolicyTrainerBase, PolicyTrainerArguments


INVALID_LOGPROB = 1.0


@dataclass
class PPOConfig(PolicyTrainerArguments):
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


class PPOTrainer(PolicyTrainerBase):

    def push_to_hub(self, **kwargs):
        """Modified from `Trainer.save_model` to only save the policy a1nd not the value network."""
        self.backup_model = self.model
        self.model = self.accelerator.unwrap_model(self.model).policy  # save only the policy
        super().push_to_hub(**kwargs)
        self.model = self.backup_model

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Modified from `Trainer.save_model` to only save the policy and not the value network."""
        # PR TODO: can we simplify this?
        # PR TODO:
        if not _internal_call:  # `push_to_hub` already swaps out the self.model with policy
            self.backup_model = self.model
            self.model = self.accelerator.unwrap_model(self.model).policy  # save only the policy
        if output_dir is None:
            output_dir = self.args.output_dir
        state_dict = self.accelerator.get_state_dict(self.backup_model)
        policy_state_dict = state_dict
        if self.accelerator.is_main_process:
            policy_state_dict = OrderedDict({k[len("policy."):]: v for k, v in state_dict.items() if k.startswith("policy.")})
        if self.args.should_save:
            self._save(output_dir, state_dict=policy_state_dict)
        if not _internal_call:
            self.model = self.backup_model

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        PR TODO: appropriate docstring
        """

        # load inputs created by the generation model in generate_batch_extras()
        queries = inputs["queries"].to(self.accelerator.device)
        query_responses = inputs["query_responses"].to(self.accelerator.device)
        responses = inputs["responses"].to(self.accelerator.device)
        gen_logprobs = inputs["generation_logprobs"].to(self.accelerator.device)
        context_length = queries.shape[1]

        with torch.no_grad():

            with self.ref_model_mgr as ref_model:
                _, ref_logprobs = self.calc_logprobs(
                    ref_model, query_responses, context_length
                )

            # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
            postprocessed_responses = responses
            if self.args.truncate_token_id:
                postprocessed_responses = self.truncate_response(responses)

            # Response Processing 2. run reward model on the truncated responses
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            sequence_lengths = self.first_true_indices(
                postprocessed_responses == self.tokenizer.pad_token_id
            ) - 1

            full_values, _, _ = self.get_reward(
                self.accelerator.unwrap_model(self.model).value_model,
                query_responses,
                context_length
            )
            values = full_values[:, context_length - 1: -1].squeeze(-1)
            _, scores, _ = self.get_reward(
                self.reward_model,
                postprocessed_query_responses,
                context_length
            )

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
            # responses not passing that filter will receive a low (fixed) score
            if self.args.non_eos_penalty:
                contain_eos_token = torch.any(postprocessed_responses == self.tokenizer.eos_token_id, dim=-1)
                non_eos_penalty_rewards = torch.full_like(scores, self.args.penalty_reward_value)
                scores = torch.where(contain_eos_token, scores, non_eos_penalty_rewards)

            # be very careful with `padding_mask`;
            # see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(
                responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)

            sequence_lengths_p1 = sequence_lengths + 1
            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
            values = torch.masked_fill(values, padding_mask_p1, 0)

            gen_logprobs = torch.masked_fill(gen_logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

            # 4. compute rewards
            kl = gen_logprobs - ref_logprobs
            non_score_reward = -self.args.kl_coef * kl
            rewards = non_score_reward.clone()
            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
            rewards[[actual_start, actual_end]] += scores

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

        # calculate gradients and loss
        output, vpred_temp = self.forward(self.model, query_responses)
        logits = output.logits[:, context_length - 1: -1]
        logits /= self.args.temperature + 1e-7
        new_all_logprobs = F.log_softmax(logits, dim=-1)
        new_logprobs = torch.gather(new_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
        new_logprobs = torch.masked_fill(
            new_logprobs, padding_mask, INVALID_LOGPROB
        )
        vpred = vpred_temp[:, context_length - 1: -1].squeeze(-1)
        vpred = torch.masked_fill(vpred, padding_mask_p1, 0)
        vpredclipped = torch.clamp(
            vpred,
            values - self.args.cliprange_value,
            values + self.args.cliprange_value,
        )

        vf_losses1 = torch.square(vpred - returns)
        vf_losses2 = torch.square(vpredclipped - returns)
        vf_loss_max = torch.max(vf_losses1, vf_losses2)
        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1)
        vf_clipfrac = masked_mean(
            (vf_losses2 > vf_losses1).float(), ~padding_mask_p1
        )
        logprobs_diff = new_logprobs - gen_logprobs
        ratio = torch.exp(logprobs_diff)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.args.cliprange,
            1.0 + self.args.cliprange
        )
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = masked_mean(pg_loss_max, ~padding_mask)
        pg_clipfrac = masked_mean(
            (pg_losses2 > pg_losses).float(), ~padding_mask
        )
        loss = pg_loss + self.args.vf_coef * vf_loss

        # calculate metrics
        with torch.no_grad():
            mean_non_score_reward = non_score_reward.sum(1).mean()

            metrics = {
                "objective/kl": kl.sum(1).mean(),
                "objective/entropy": (-gen_logprobs).sum(1).mean(),
                "objective/non_score_reward": non_score_reward.mean(),
                "objective/rlhf_reward": mean_non_score_reward + scores.mean(),
                "objective/scores": self.accelerator.gather(scores.mean()).mean().item(),
                "policy/approxkl_avg": 0.5 * (logprobs_diff**2).mean(),
                "policy/clipfrac_avg": pg_clipfrac.mean(),
                "loss/policy_avg": self.accelerator.gather(pg_loss).mean().item(),
                "loss/value_avg": vf_loss.mean(),
                "val/clipfrac_avg": vf_clipfrac.mean(),
                # "policy/entropy_avg":
                # "val/ratio":
                # "val/ratio_var":
                "val/num_eos_tokens": (responses == self.tokenizer.eos_token_id).sum().item(),
            }

        return loss, metrics

    def generate_completions(self, sampling: bool = False):
        # PR TODO: move this to eval step maybe?
        """for eval"""
        args = self.args
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
            name = f"trained {args.base_model}"
            with torch.no_grad():
                context_length = query.shape[1]
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    query_response, _ = self.generate(
                        unwrapped_model.policy,
                        query,
                        generation_config,
                    )
                response = query_response[:, context_length:]
                postprocessed_response = response
                if args.truncate_token_id:
                    postprocessed_response = self.truncate_response(args, self.tokenizer, response)
                table["query"].extend(gather_object(self.tokenizer.batch_decode(query, skip_special_tokens=True)))
                table[name].extend(gather_object(self.tokenizer.batch_decode(postprocessed_response)))

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                _, score, _ = self.get_reward(self.reward_model, postprocessed_query_response, context_length)
                table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)
        # PR TODO: write df to pickle if not using wandb
        if "wandb" in args.report_to:
            import wandb
            if wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})
