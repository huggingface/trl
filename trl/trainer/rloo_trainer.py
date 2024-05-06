from dataclasses import dataclass
from typing import Dict, Tuple, Union, Any
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from . import PolicyTrainerBase, PolicyTrainerArguments


INVALID_LOGPROB = 1.0


@dataclass
class RLOOConfig(PolicyTrainerArguments):
    cliprange: float = 0.2
    """the clip range"""
    kl_coef: float = 0.10
    """the KL coefficient"""
    rloo_k: int = 2
    """REINFORCE Leave-One-Out (RLOO) number of online samples per prompt"""


class RLOOTrainer(PolicyTrainerBase):
    _tag_names = ["trl", "rloo"]

    def generate_batch_extras(self, model, input_ids):
        input_ids = input_ids.repeat(self.args.rloo_k, 1)
        return super().generate_batch_extras(model, input_ids)

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
            gen_logprobs = torch.masked_fill(gen_logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

            # 4. compute rewards
            kl = gen_logprobs - ref_logprobs
            non_score_reward = (-self.args.kl_coef * kl).sum(1)
            rlhf_reward = scores + non_score_reward.unsqueeze(1)

            # We generated `self.args.rloo_k` responses per prompt
            # RLOO loss: reward minus avg rewards of other responses
            rlhf_sum = rlhf_reward.sum(dim=0, keepdim=True)
            n = rlhf_reward.size(0)
            mean_other = (rlhf_sum - rlhf_reward) / (n - 1)
            advantages = rlhf_reward - mean_other

        # calculate gradients and loss
        active_logits, active_logprobs = self.calc_logprobs(
            model, query_responses, context_length
        )
        active_logprobs = torch.masked_fill(
            active_logprobs, padding_mask, INVALID_LOGPROB
        )
        new_ratio = (active_logprobs - gen_logprobs).exp()
        logprobs_diff = active_logprobs.sum(1) - gen_logprobs.sum(1)
        ratio = torch.exp(logprobs_diff)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = pg_loss_max.mean()
        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()

        # get metrics
        with torch.no_grad():
            prob_dist = torch.nn.functional.softmax(active_logits, dim=-1)
            entropy_avg = (
                torch.logsumexp(active_logits, dim=-1) -
                torch.sum(prob_dist * active_logits, dim=-1)
            ).mean()
            metrics = {
                "objective/kl": kl.sum(1).mean(),
                "objective/entropy": (-gen_logprobs).sum(1).mean(),
                "objective/rlhf_reward": rlhf_reward.mean(),
                "objective/scores": self.accelerator.gather(scores.mean()).mean().item(),
                "policy/approxkl_avg": 0.5 * (logprobs_diff**2).mean(),
                "loss/policy_avg": self.accelerator.gather(pg_loss).mean().item(),
                "val/ratio": new_ratio.mean(),
                # "val/ratio_var": new_ratio.mean().var(),
                "val/num_eos_tokens": (responses == self.tokenizer.eos_token_id).sum().item(),
                "policy/clipfrac_avg": self.accelerator.gather(pg_clipfrac).mean().item(),
                "policy/entropy_avg": entropy_avg
            }

        return (pg_loss, metrics)
