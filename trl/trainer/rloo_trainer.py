from copy import deepcopy
import gc
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable, Any
import warnings
from contextlib import nullcontext

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
    PreTrainedTokenizerBase
)

from ..core import logprobs_from_logits

from ..models import SUPPORTED_ARCHITECTURES, create_reference_model, PreTrainedModelWrapper

from . import PolicyTrainerBase, PolicyTrainerArguments

from ..import_utils import is_peft_available


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

    def calc_logprobs(self, model, query_responses, context_length):
        responses = query_responses[:, context_length:]
        output_logits = self.forward(model, query_responses).logits
        response_logits = output_logits[:, context_length - 1 : -1]
        response_logits /= max(self.args.temperature, 1e-7)
        response_logprobs = logprobs_from_logits(response_logits, responses, gather=True)
        return response_logits, response_logprobs


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/trainer.py#L3112
        """
        queries = inputs["queries"].to(self.accelerator.device)
        context_length = queries.shape[1]
        query_responses = inputs["query_responses"].to(self.accelerator.device)
        responses = inputs["responses"].to(self.accelerator.device)
        gen_logprobs = inputs["generation_logprobs"].to(self.accelerator.device)
        with torch.no_grad(), self.time_metric_ctx("calc_advantages"):
            # PR TODO: refactor into a function shared by ppov2 which calculates sequences and logprobs
            #          see DPOTrainer.concatenated_forward

            with self.cast_model_ctx(), self.ref_model_mgr as ref_model:
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

            # clear cache before get_reward call
            gc.collect()
            torch.cuda.empty_cache()

            with self.time_metric_ctx("get_reward"):
                _, scores, _ = self.get_reward(
                    self.reward_model,
                    postprocessed_query_responses,
                    context_length
                )
            gc.collect()
            torch.cuda.empty_cache()

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            if self.args.non_eos_penalty:
                contain_eos_token = torch.any(postprocessed_responses == self.tokenizer.eos_token_id, dim=-1)
                scores = torch.where(contain_eos_token, scores, torch.full_like(scores, self.args.penalty_reward_value))
                # PR TODO: remove this debug statement
                self.accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

            # be very careful with `padding_mask`;
            # see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            gen_logprobs = torch.masked_fill(gen_logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

            # 4. compute rewards
            kl = gen_logprobs - ref_logprobs
            non_score_reward = (-self.args.kl_coef * kl).sum(1)
            rlhf_reward = scores + non_score_reward.unsqueeze(1)

            # we generated `self.args.rloo_k` many responses per prompt
            # now we can implement the RLOO loss by subtracting the reward of
            # a response by the average rewards of other `rloo_k - 1` responses
            rlhf_sum = rlhf_reward.sum(dim=0, keepdim=True)
            n = rlhf_reward.size(0)
            mean_other = (rlhf_sum - rlhf_reward) / (n - 1)
            advantages = rlhf_reward - mean_other

        # calculate gradients and loss
        with self.time_metric_ctx("calc_loss"):

            with self.cast_model_ctx():
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

        # log metrics
        with torch.no_grad():
            prob_dist = torch.nn.functional.softmax(active_logits, dim=-1)
            entropy_avg = (
                torch.logsumexp(active_logits, dim=-1)
                - torch.sum(prob_dist * active_logits, dim=-1)
            ).mean()

            self.store_metrics({
                "objective/kl": kl.sum(1).mean(),
                "objective/entropy": (-gen_logprobs).sum(1).mean(),
                "objective/rlhf_reward": rlhf_reward.mean(),
                "objective/scores": self.accelerator.gather(scores.mean()).mean().item(),
                "policy/approxkl_avg": 0.5 * (logprobs_diff**2).mean(),
                "loss/policy_avg": self.accelerator.gather(pg_loss).mean().item(),
                "val/ratio": new_ratio.mean(),
                #"val/ratio_var": new_ratio.mean().var(),
                "val/num_eos_tokens": (responses == self.tokenizer.eos_token_id).sum().item(),
                "policy/clipfrac_avg": self.accelerator.gather(pg_clipfrac).mean().item(),
                "policy/entropy_avg": entropy_avg
            })

        loss = pg_loss.to(self.args.device)

        # PR TODO: decorator which calls gc.collect(); torch.cuda.empty_cache()

        if return_outputs:
            return (loss, metrics)
        return loss

if __name__ == "__main__":
    pass
