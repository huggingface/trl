from copy import deepcopy
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


from ..import_utils import is_peft_available

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


INVALID_LOGPROB = 1.0


@dataclass
class RLOOConfig(TrainingArguments):
    response_length: int = 53
    """the length of the response"""
    truncate_token: Optional[Literal["eos"]] = None
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = False
    """whether to penalize responses that do not contain `truncate_token_id`"""

    cliprange: float = 0.2
    """the clip range"""
    kl_coef: float = 0.05
    """the KL coefficient"""

    # rloo args
    rloo_k: int = 4
    """REINFORCE Leave-One-Out (RLOO) number of online samples per prompt"""


"""
PR TODO: class ModelWithRewardsConfig(ModelConfig)
- reward_model_path
- sft_model_path
"""


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def get_reward_model_reward(reward_model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(reward_model, reward_model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    reward_logits = reward_model.score(output.hidden_states[-1])
    sequence_lengths = (
        first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    )
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return reward_logits[
        torch.arange(reward_logits.size(0), device=reward_logits.device),
        sequence_lengths,
    ].squeeze(-1),


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[0]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


class RLOOTrainer(Trainer):
    def __init__(
            self,
            model: Optional[PreTrainedModelWrapper],
            args: RLOOConfig,
            train_dataset: Union[Dataset, "datasets.Dataset"],
            reward_model: Optional[PreTrainedModelWrapper] = None,
            reward_fn: Callable = None,
            ref_model: Optional[PreTrainedModelWrapper] = None,
            train_generation_config: Optional[GenerationConfig] = None,
            eval_generation_config: Optional[GenerationConfig] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            **kwargs
    ) -> None:

        self.ref_model = ref_model
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        assert (reward_model is not None) != (reward_fn is not None), "Must set either reward_model or reward_fn, but not both"
        self.reward_model = reward_model
        self.reward_fn = reward_fn


        default_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        self.train_generation_config = train_generation_config or default_generation_config
        self.eval_generation_config = eval_generation_config or default_generation_config
        # disable `pad_token_id` and `eos_token_id` because we just want to
        # generate tokens without truncation / padding
        self.train_generation_config.eos_token_id = None
        self.train_generation_config.pad_eos_token_id = None


        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )

        self._prepare_multigpu()

        #########################
        # Prepare reference model
        #########################
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(self.model)
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None` "
                f"got {type(ref_model)} "
                f"- supported architectures are: {SUPPORTED_ARCHITECTURES} "
            )
        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
            if self.is_peft_model
            else nullcontext
        )

        #########
        # disable dropout
        #########
        # PR TODO: review the below, I'm not sure why we disable dropout
        disable_dropout(self.model)
        if self.ref_model is not None:
            disable_dropout(self.ref_model)
        if self.reward_model is not None:
            disable_dropout(self.reward_model)


    def _prepare_multigpu(self):
        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.ref_model is None:
            return
        elif self.is_deepspeed_enabled:
            self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare_model(
                self.ref_model,
                evaluation_mode=True
            )

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/trainer.py#L3112
        """
        model.train()

        inputs = self._prepare_inputs(inputs)
        queries = inputs["input_ids"].to(self.accelerator.device)
        queries = queries.repeat(self.args.rloo_k, 1)

        context_length = queries.shape[1]
        group_query_response, group_logits = generate(
            self.accelerator.unwrap_model(model),
            queries,
            self.tokenizer,
            self.train_generation_config,
        )
        with torch.no_grad(), self.optional_peft_ctx():
            group_ref_output = forward(self.ref_model, group_query_response, self.tokenizer)
            group_ref_output_logits = group_ref_output.logits

        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        scores = []
        sequence_lengths = []
        for query, query_response, logits, ref_output_logits in zip(queries, group_query_response, group_logits, group_ref_output_logits):

            response = query_response[context_length:]

            # use the logits during generation directly, instead of using the following
            all_logprob = F.log_softmax(logits, dim=-1)
            print("query_response", query_response.shape)
            print("response", response.shape)
            print("logits", logits.shape)
            print("all_logprob", all_logprob.shape)
            logprob = torch.gather(all_logprob, -1, response.unsqueeze(-1)).squeeze(-1)
            del logits, all_logprob
            torch.cuda.empty_cache()

            ref_logits = ref_output_logits[:, context_length - 1 : -1]
            ref_logits /= self.args.temperature + 1e-7
            ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
            ref_logprob = torch.gather(ref_all_logprob, -1, response.unsqueeze(-1)).squeeze(-1)
            del ref_output_logits, ref_logits, ref_all_logprob
            torch.cuda.empty_cache()

            # Response Processing 1. truncate response after the
            # first occurrence of `truncate_token_id`
            postprocessed_response = response
            if self.args.truncate_token_id:
                postprocessed_response = truncate_response(self.args, self.tokenizer, response)

            # Response Processing 2. run reward model on the truncated responses
            postprocessed_query_response = torch.cat((query, postprocessed_response), 0)
            sequence_length = first_true_indices(postprocessed_response == self.tokenizer.pad_token_id) - 1
            if self.reward_model:
                score = get_reward_model_reward(
                    self.reward_model,
                    postprocessed_query_response,
                    self.tokenizer,
                    context_length
                )
            else:
                self.reward_fn(
                    postprocessed_query_response,
                    self.tokenizer,
                    context_length
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
        postprocessed_responses = torch.cat(postprocessed_responses, 0)
        logprobs = torch.cat(logprobs, 0)
        ref_logprobs = torch.cat(ref_logprobs, 0)
        print(f"{(ref_logprobs - logprobs).exp()=}")
        sequence_lengths = torch.cat(sequence_lengths, 0)
        scores = torch.cat(scores, 0)
        del (logprob, ref_logprob, score)
        torch.cuda.empty_cache()

        # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
        # responses not passing that filter will receive a low (fixed) score
        # only query humans on responses that pass that filter
        contain_eos_token = torch.any(postprocessed_responses == self.tokenizer.eos_token_id, dim=-1)
        if self.args.non_eos_penalty:
            scores = torch.where(contain_eos_token, scores, torch.full_like(scores, self.args.penalty_reward_value))
        self.accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

        # be very careful with `padding_mask_p1`;
        # see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

        # 4. compute rewards
        kl = logprobs - ref_logprobs
        non_score_reward = (-self.args.kl_coef * kl).sum(1)
        rlhf_reward = scores - non_score_reward

        # we generated `self.args.rloo_k` many responses per prompt
        # now we can implement the RLOO loss by subtracting the reward of
        # a response by the average rewards of other `rloo_k - 1` responses
        advantages = torch.zeros_like(rlhf_reward)
        for i in range(0, len(advantages)):
            other_response_rlhf_rewards = []
            for j in range(0, len(advantages)):
                if i != j:
                    other_response_rlhf_rewards.append(rlhf_reward[j])
            advantages[i] = rlhf_reward[i] - torch.stack(other_response_rlhf_rewards).mean()
        torch.cuda.empty_cache()

        # calculate loss
        with self.accelerator.accumulate(model):
            output = forward(model, query_responses, self.tokenizer)
            logits = output.logits[:, context_length - 1 : -1]
            logits /= self.args.temperature + 1e-7
            new_all_logprobs = F.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(new_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            new_logprobs = torch.masked_fill(
                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
            )
            new_ratio = (new_logprobs - logprobs).exp()
            new_logprobs = new_logprobs.sum(1)
            logprobs = logprobs.sum(1)
            logprobs_diff = new_logprobs - logprobs
            ratio = torch.exp(logprobs_diff)
            # print(f"{ratio=}")
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
            pg_loss_max = torch.max(pg_losses, pg_losses2)
            pg_loss = pg_loss_max.mean()
            pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
            loss = pg_loss
            self.accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                approxkl = 0.5 * (logprobs_diff**2).mean()

        # del everything and empty cache
        # fmt: off
        del (
            output, logits, new_all_logprobs, new_logprobs,
            logprobs_diff, ratio, pg_losses, pg_losses2,
            pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
        )
        # fmt: on
        torch.cuda.empty_cache()

        with torch.no_grad():
            rlhf_reward_mean = self.accelerator.gather(rlhf_reward).mean().item()
            self.accelerator.print(f"{rlhf_reward_mean=}")
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            # PR TODO: why is this metric removed?
            # mean_non_score_reward = non_score_reward.sum(1).mean()

            self.log({
                "objective/kl": self.accelerator.gather(mean_kl).mean().item(),
                "objective/entropy": self.accelerator.gather(mean_entropy).mean().item(),
                "objective/non_score_reward": self.accelerator.gather(mean_non_score_reward).mean().item(),
                "objective/rlhf_reward": self.accelerator.gather(rlhf_reward).mean().item(),
                "objective/scores": self.accelerator.gather(scores.mean()).mean().item(),
                "policy/approxkl_avg": self.accelerator.gather(approxkl).mean().item(),
                "policy/clipfrac_avg": self.accelerator.gather(pg_clipfrac).mean().item(),
                "loss/policy_avg": self.accelerator.gather(pg_loss).mean().item(),
                "loss/value_avg": self.accelerator.gather(vf_loss_stats).mean().item(),
                "val/clipfrac_avg": self.accelerator.gather(vf_clipfrac_stats).mean().item(),
                "policy/entropy_avg": self.accelerator.gather(entropy).mean().item(),
                "val/ratio": self.accelerator.gather(new_ratio).mean().item(),
                "val/ratio_var": self.accelerator.gather(ratio_stats).var().item(),
                "val/num_eos_tokens": (responses == self.tokenizer.eos_token_id).sum().item(),
            })

        del kl, mean_kl, mean_entropy, scores
        torch.cuda.empty_cache()

        return loss.detach() / self.args.gradient_accumulation_steps


if __name__ == "__main__":
    pass
