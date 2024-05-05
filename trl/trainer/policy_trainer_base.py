from collections import defaultdict
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

from trl.models.utils import unwrap_model_for_generation, prepare_model_and_ref_model


from ..models import SUPPORTED_ARCHITECTURES, create_reference_model, PreTrainedModelWrapper
from .utils import disable_dropout_in_model, peft_module_casting_to_bf16, peft_module_casting_to_fp16


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
    temperature: float = 1.0
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = False
    """whether to penalize responses that do not contain `truncate_token_id`"""

    update_generation_model_step: Optional[int] = None
    """Number of steps between updating the generation model. If None, once per epoch"""


class fast_eval_mode:
    """
    Convert to model.eval(), then revert to previous state

    Behavior
    - DOESNT disable grad
    - Disable dropout layers
    - Freeze BatchNorm
    """
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.was_training = self.model.training
        if self.was_training:
            self.model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_training:
            self.model.train()


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

# PR TODO: Implement original workflow as follows
-"""
-There are three models forwards() considered in each step
-- ref_model: Never changes
-- update_model: changes every update step (originally was after multiple epochs, I expect fractions of an epoch to work better. Will have to validate to ensure this is true.)
-- active_model: changes every step automatically via trainer.py
-
-Caveat: We will specify update_generation_model_steps which will adjust the generation model some fraction or multiple of the number of epochs
ppp-
-generation only uses update_model
-
-- while iterating over steps for num_train_epochs
-- using callback: if update_generation_model_steps set the `update_model` adapter to a copy of `active_model` adapter
-- for each step
-  - generate with update_model
-  - apply forward pass with ref_model, update_model, and active_model
-

-Work
-- 0) Implement ModelManager
-- 1) Implement Callback classes to sync update_model (add comment that we can augment this to prefill with generated samples, but by default generate at runtime)
-- 2) Update rloo_trainer.py to use the three models
-- Important: Design with consideration that the generation step could take place within vllm
-  - this can be done with a "get_generated" call, which makes an easy in-place option for retrieving pre-calculated
"""


class UpdateTrainGenerationSamplesCallback(TrainerCallback):
    def __init__(self, model, steps, batch_size):
        self.model = model
        self.steps = steps
        self.batch_size = batch_size

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.steps == 0:
            dataloader = kwargs['dataloader']
            for i, batch in enumerate(dataloader):
                inputs = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = self.model.generate(inputs, attention_mask=attention_mask)
                queries = self.model.tokenizer.batch_decode(inputs, skip_special_tokens=True)
                query_responses = self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                context_lengths = attention_mask.sum(dim=1).tolist()
                generation_logprobs = self.model(**batch).logits.max(dim=-1)[0].detach().numpy().tolist()

                # Modify the dataset with new fields
                batch.update({
                    'queries': queries,
                    'query_responses': query_responses,
                    'context_length': context_lengths,
                    'generation_logprobs': generation_logprobs
                })

                # Example of how to replace the batch in the dataloader, this will depend on your DataLoader setup
                dataloader.dataset[i] = batch

                if i >= self.batch_size:
                    break


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
        self.optional_peft_ctx = None
        if self.ref_model is not None:
            return self.ref_model
        elif self.is_peft_model:
            self.optional_peft_ctx = self.accelerator.unwrap_model(self.model).disable_adapter()
            return self.model
        else:
            raise ValueError

    def __exit__(self, exc_type, exc_value, traceback):
        if self.optional_peft_ctx is not None:
            with self.optional_peft_ctx:
                pass  # teardown


class PolicyTrainerBase(Trainer):
    """
    Base class for implementing a policy training algorithm.
    # PR TODO: document arguments
    """
    def __init__(
            self,
            model: Optional[PreTrainedModelWrapper],
            ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[TrainingArguments] = None,
            train_dataset: Optional[Dataset] = None,
            reward_model: Optional[PreTrainedModelWrapper] = None,
            reward_fn: Callable = None,
            train_generation_config: Optional[GenerationConfig] = None,
            eval_generation_config: Optional[GenerationConfig] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init_kwargs: Optional[Dict] = None,
            ref_model_init_kwargs: Optional[Dict] = None,
            peft_config: Optional = None,
            force_use_ref_model: bool = False,
            **kwargs
    ) -> None:

        model, ref_model = prepare_model_and_ref_model(
            model=model,
            ref_model=ref_model,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            peft_config=peft_config,
            force_use_ref_model=force_use_ref_model,
            args=args,
        )

        # PR TODO: class variable which determines whether ref logprobs are generated either
        #          - once per batch
        #          - once per update
        #          - once per run (see dpo_trainer.py precompute_ref_log_probs)

        # Disable dropout ensures logprobs during generation aren't different from forward pass
        # https://github.com/huggingface/trl/pull/1586#discussion_r1579533825
        for m in [model, ref_model, reward_model]:
            if m is not None:
                disable_dropout_in_model(m)

        # PR TODO: subclass with RewardTrainerBase which accepts a reward_model or reward_fn
        #          remove the below from this class
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


        # handle casting self.model
        if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
            self.cast_model_ctx = lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            self.cast_model_ctx = nullcontext

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Add callback to update generation model
        update_train_gen_cb = UpdateTrainGenerationSamplesCallback(
            model=self.model,
            steps=(
                self.args.update_generation_step or
                len(self.get_train_dataloader()) / self.args.gradient_accumulation_steps
            ),
            batch_size=self._train_batch_size
        )
        self.add_callback(update_train_gen_cb)

        self.ref_model_mgr = ReferenceModelManager(
            self.accelerator,
            ref_model=ref_model,
            model=model,
            is_deepspeed_enabled=self.is_deepspeed_enabled,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)


        # PR TOOD: accelerate with reward model
        #self.reward_model.to(self.accelerator.device)
        #self.reward_model = _prepare_multigpu(
        #    self.reward_model,
        #    self.accelerator,
        #    self.is_deepspeed_enabled
        #)

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    @staticmethod
    def _disable_dropout(model):
        if model is None:
            return
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0

    def generate(self, model, queries, generation_config, requires_grad=True):
        """generate in a way that does not affect padding tokens"""
        if True:  # with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model = model
            with (fast_eval_mode(unwrapped_model) if requires_grad else nullcontext):
                context_length = queries.shape[1]
                attention_mask = queries != self.tokenizer.pad_token_id
                input_ids = torch.masked_fill(queries, ~attention_mask, 0)
                output = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    # PR TODO: https://github.com/huggingface/trl/pull/1540/files#r1588004580
                    output_logits=True,
                )
        logits = torch.stack(output.logits, 1)
        query_responses = torch.cat((queries, output.sequences[:, context_length:]), dim=1)
        return query_responses, logits

    def forward(self, model, query_responses):
        attention_mask = query_responses != self.tokenizer.pad_token_id
        # position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        #with eval_mode(model):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )

    def get_reward(self, reward_model, query_responses, context_length):
        attention_mask = query_responses != self.tokenizer.pad_token_id

        # PR TODO: figure out why we had to get base_model_prefix
        # lm_backbone = getattr(reward_model, reward_model.base_model_prefix)

        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

        #with eval_mode(reward_model):
        output = reward_model(
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
        trunc_idxs = self.first_true_indices(responses == self.args.truncate_token_id).unsqueeze(-1)
        new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
        idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
        postprocessed_responses = torch.masked_fill(
            responses, idxs > trunc_idxs,
            self.tokenizer.pad_token_id
        )
        return postprocessed_responses


    @staticmethod
    def first_true_indices(bools, dtype=torch.long):
        """
        Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
        the position of the first True in each "row".

        Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
        """
        row_len = bools.size(-1)
        zero_or_index = (
            row_len * (~bools).type(dtype)
            + torch.arange(row_len, dtype=dtype, device=bools.device)
        )
        return torch.min(zero_or_index, dim=-1).values


    def store_metrics(
            self,
            metrics: Dict[str, float],
            train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).to(dtype=torch.float32).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    def time_metric_ctx(self, timer_name: str):
        from time import perf_counter
        timer_metric_name = f"timer/{timer_name}"
        class catchtime:
            def __enter__(s):
                s.start = perf_counter()
            def __exit__(s, type, value, traceback):
                runtime = perf_counter() - s.start
                self.store_metrics({timer_metric_name: runtime})
        return catchtime()


    def training_step(self, *args, **kwargs):
        """time logged training step"""
        with self.time_metric_ctx("training_step"):
            return super().training_step(*args, **kwargs)



if __name__ == "__main__":
    pass
