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

from trl.models.utils import unwrap_model_for_generation


from ..models import SUPPORTED_ARCHITECTURES, create_reference_model, PreTrainedModelWrapper
from .utils import disable_dropout_in_model

from ..import_utils import is_peft_available

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


@dataclass
class PolicyTrainerArguments(TrainingArguments):
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

"""
PR TODO: class ModelWithRewardsConfig(ModelConfig)
- reward_model_path
- sft_model_path
"""


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


# PR TODO: maybe this isn't necessary? This may be handled already by the accelerator, as it prepares
#          any model type object in Accelerator._prepare_deepspeed
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/accelerator.py#L1530
def _prepare_deepspeed(self, accelerator, model: PreTrainedModelWrapper, evaluation_mode):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
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
    if evaluation_mode:
        model.eval()
    return model


def _prepare_multigpu(model, accelerator, is_deepspeed_enabled: bool):
    if model is None:
        return model
    elif is_deepspeed_enabled:
        return _prepare_deepspeed(
            accelerator,
            model,
            evaluation_mode=True
        )
    else:
        return accelerator.prepare_model(
            model,
            evaluation_mode=True
        )


class ReferenceModelManager:
    """
    Context manager to prepare and manage the reference model.
    - If it doesn't exist create a reference model
      - OR use the base model with adapters disabled if base model uses PEFT
    - Distribute the model to the accelerator
    """
    def __init__(
            self,
            accelerator,
            ref_model: Optional[PreTrainedModelWrapper] = None,
            model: Optional[PreTrainedModelWrapper] = None,
            is_deepspeed_enabled: bool = False
    ):
        self.accelerator = accelerator
        if not is_peft_available():
            self.is_peft_model = False
        else:
            self.is_peft_model = (
                getattr(model, "is_peft_model", False)
                or isinstance(model, PeftModel)
            )

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(model)
        elif self.is_peft_model:
            self.ref_model = None
            self.model = model
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None` "
                f"got {type(ref_model)} "
                f"- supported architectures are: {SUPPORTED_ARCHITECTURES} "
            )

        if self.ref_model is not None and not self.is_peft_model:
            print(type(self.ref_model))
            self.ref_model = _prepare_multigpu(self.ref_model, self.accelerator, is_deepspeed_enabled)

    def __enter__(self):
        if self.ref_model is not None:
            return self.ref_model
        elif self.is_peft_model:
            return self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
        else:
            raise ValueError

    def __exit__(self, exc_type, exc_value, traceback):
        if self.ref_model is None and self.is_peft_model:
            self.optional_peft_ctx.__exit__(exc_type, exc_value, traceback)



# PR TODO: determine why disable_dropout existed, and if it's necessary, readd it
# https://github.com/huggingface/trl/pull/1540/files/c54f111836a0e8b3af2fd6338a2decbe74b7d494#r1580194071


class PolicyTrainerBase(Trainer):
    def __init__(
            self,
            model: Optional[PreTrainedModelWrapper],
            args: TrainingArguments,
            train_dataset: Union[Dataset, "datasets.Dataset"],
            reward_model: Optional[PreTrainedModelWrapper] = None,
            reward_fn: Callable = None,
            ref_model: Optional[PreTrainedModelWrapper] = None,
            train_generation_config: Optional[GenerationConfig] = None,
            eval_generation_config: Optional[GenerationConfig] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            **kwargs
    ) -> None:

        # Disable dropout ensures logprobs during generation aren't different from forward pass
        # https://github.com/huggingface/trl/pull/1586#discussion_r1579533825
        for m in [model, ref_model, reward_model]:
            if m is not None:
                disable_dropout_in_model(m)

        assert (reward_model is not None) != (reward_fn is not None), "Must set either reward_model or reward_fn, but not both"
        if reward_model is not None and "score" not in dir(reward_model):
            raise TypeError(f"Reward model of type {type(reward_model)} has no score function.")
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
        if False:
            # PR TODO: review this??
            self.train_generation_config.eos_token_id = None
            self.train_generation_config.pad_token_id = None
        else:
            self.train_generation_config.eos_token_id = tokenizer.eos_token_id
            self.train_generation_config.pad_token_id = tokenizer.pad_token_id


        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )

        self.ref_model_mgr = ReferenceModelManager(
            self.accelerator,
            ref_model=ref_model,
            model=model,
            is_deepspeed_enabled=self.is_deepspeed_enabled,
        )

        # PR TOOD: accelerate with reward model
        #self.reward_model.to(self.accelerator.device)
        #self.reward_model = _prepare_multigpu(
        #    self.reward_model,
        #    self.accelerator,
        #    self.is_deepspeed_enabled
        #)

    @staticmethod
    def _disable_dropout(model):
        if model is None:
            return
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0

    def generate(self, model, queries, generation_config):
        """generate in a way that does not affect padding tokens"""
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            context_length = queries.shape[1]
            attention_mask = queries != self.tokenizer.pad_token_id
            input_ids = torch.masked_fill(queries, ~attention_mask, 0)
            output = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
                # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        logits = torch.stack(output.scores, 1)
        query_responses = torch.cat((queries, output.sequences[:, context_length:]), dim=1)
        return query_responses, logits

    def forward(self, model, query_responses):
        attention_mask = query_responses != self.tokenizer.pad_token_id
        # position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )

    def get_reward(self, reward_model, query_responses, context_length):
        attention_mask = query_responses != self.tokenizer.pad_token_id
        lm_backbone = getattr(reward_model, reward_model.base_model_prefix)
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        output = lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        reward_logits = reward_model.score(output.hidden_states[-1])
        sequence_lengths = (
            first_true_indices(
                query_responses[:, context_length:] == self.tokenizer.pad_token_id
            ) - 1 + context_length
        )

        # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
        return (
            reward_logits,
            reward_logits[
                torch.arange(reward_logits.size(0), device=reward_logits.device),
                sequence_lengths,
            ].squeeze(-1),
            sequence_lengths,
        )

    def truncate_response(self, responses):
        trunc_idxs = first_true_indices(responses == self.args.truncate_token_id).unsqueeze(-1)
        new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
        idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
        postprocessed_responses = torch.masked_fill(
            responses, idxs > trunc_idxs,
            self.tokenizer.pad_token_id
        )
        return postprocessed_responses


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    pass
