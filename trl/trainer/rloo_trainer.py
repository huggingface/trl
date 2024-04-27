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


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


class eval_mode:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.was_training = self.model.training
        if self.was_training:
            self.model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_training:
            self.model.train()


class RLOOTrainer(PolicyTrainerBase):

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
                with eval_mode(ref_model) as _ref_model:
                    ref_output_logits = self.forward(_ref_model, query_responses).logits
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

        # clear cache before get_reward call
        gc.collect()
        torch.cuda.empty_cache()

        with eval_mode(self.reward_model) as _reward_model:
            _, scores, _ = self.get_reward(
                _reward_model,
                postprocessed_query_responses,
                context_length
            )
        torch.cuda.empty_cache()

        # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
        # responses not passing that filter will receive a low (fixed) score
        # only query humans on responses that pass that filter
        contain_eos_token = torch.any(postprocessed_responses == self.tokenizer.eos_token_id, dim=-1)
        if self.args.non_eos_penalty:1
            scores = torch.where(contain_eos_token, scores, torch.full_like(scores, self.args.penalty_reward_value))
        # PR TODO: this is from original, but maybe it should be logged somewhere?
        #self.accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

        # be very careful with `padding_mask`;
        # see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

        # 4. compute rewards
        kl = logprobs - ref_logprobs
        non_score_reward = (-self.args.kl_coef * kl).sum(1)
        rlhf_reward = scores + non_score_reward.unsqueeze(1)

        # we generated `self.args.rloo_k` many responses per prompt
        # now we can implement the RLOO loss by subtracting the reward of
        # a response by the average rewards of other `rloo_k - 1` responses
        advantages = torch.zeros_like(rlhf_reward)
        for i in range(0, len(advantages)):
            other_response_rlhf_rewards = []
            for j in range(0, len(advantages)):
                if i != j:
                    other_response_rlhf_rewards.append(rlhf_reward[j])
            advantages[i] = rlhf_reward[i] - torch.stack(other_response_rlhf_rewards).mean(0)
        torch.cuda.empty_cache()

        # calculate loss
        output = self.forward(model, query_responses)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= self.args.temperature + 1e-7
        new_all_logprobs = F.log_softmax(logits, dim=-1)
        new_logprobs = torch.gather(new_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
        new_logprobs = torch.masked_fill(
            new_logprobs, padding_mask, INVALID_LOGPROB
        )
        new_ratio = (new_logprobs - logprobs).exp()
        logprobs_diff = new_logprobs.sum(1) - logprobs.sum(1)
        ratio = torch.exp(logprobs_diff)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = pg_loss_max.mean()
        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()

        # backprop
        with self._cast_base_model_ctx():
            self.accelerator.backward(pg_loss)

        # log metrics
        with torch.no_grad():
            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
            approxkl = 0.5 * (logprobs_diff**2).mean()

            rlhf_reward_mean = self.accelerator.gather(rlhf_reward).mean().item()
            # PR TODO: this is from original, but maybe it should be logged somewhere?
            #self.accelerator.print(f"{rlhf_reward_mean=}")
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            # PR TODO: why is this metric removed in the original
            # mean_non_score_reward = non_score_reward.sum(1).mean()
            # "objective/non_score_reward"

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

        loss = pg_loss.detach()

        del (
            output, logits, new_all_logprobs, new_logprobs,
            logprobs_diff, ratio, pg_losses, pg_losses2,
            pg_loss, pg_clipfrac, prob_dist, entropy, approxkl,
            kl, mean_kl, mean_entropy, scores
        )
        torch.cuda.empty_cache()

        return loss


if __name__ == "__main__":
    pass
