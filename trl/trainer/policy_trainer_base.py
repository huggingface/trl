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


class ReferenceModelManager:
    def __init__(
            self,
            accelerator,
            ref_model: Optional[PreTrainedModelWrapper] = None,
            model: Optional[PreTrainedModelWrapper] = None,
            is_deepspeed_enabled: bool = False
    ):
        self.accelerator = accelerator
        self.is_peft_model = getattr(model, "is_peft_model", False)

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
            self.ref_model.to(self.accelerator.device)
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(model)
            self.ref_model.to(self.accelerator.device)
        elif self.is_peft_model:
            self.ref_model = None
            self.model = model
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None` "
                f"got {type(ref_model)} "
                f"- supported architectures are: {SUPPORTED_ARCHITECTURES} "
            )

        self._prepare_multigpu(is_deepspeed_enabled)

    def _prepare_multigpu(self, is_deepspeed_enabled: bool):
        if self.ref_model is None:
            return
        elif is_deepspeed_enabled:
            self.ref_model = _prepare_deepspeed(
                self.accelerator,
                self.ref_model,
                evaluation_mode=True
            )
        else:
            self.ref_model = self.accelerator.prepare_model(
                self.ref_model,
                evaluation_mode=True
            )

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
        self.train_generation_config.eos_token_id = None
        self.train_generation_config.pad_token_id = None


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

        # PR TODO: what about multi-gpu here? Shouldn't we _prepare_multigpu(reward_model) as well?
        self.reward_model.to(self.model.device)

    def generate(self, lm_backbone, queries, generation_config):
        """generate in a way that does not affect padding tokens"""
        context_length = queries.shape[0]
        print("queries", queries)
        print("queries.shape", queries.shape)
        attention_mask = queries != self.tokenizer.pad_token_id
        print("attention_mask", attention_mask)
        print("attention_mask.shape", attention_mask.shape)
        input_ids = torch.masked_fill(queries, ~attention_mask, 0)
        print("input_ids", input_ids)
        print("input_ids.shape", input_ids.shape)
        output = lm_backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
            # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
        print("output", output)
        logits = torch.stack(output.scores, 1)
        print("logits", logits)
        print("logits.shape", logits.shape)

        query_responses = torch.cat((queries, output.sequences[:, context_length:]), dim=1)
        print("query_responses", query_responses)
        print("query_responses.shape", query_responses.shape)

        return query_responses, logits

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    pass
