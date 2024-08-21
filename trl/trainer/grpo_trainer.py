# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from ..core import (
    WANDB_PADDING,
    GRPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
from ..import_utils import is_npu_available, is_torch_greater_2_0, is_xpu_available
from ..models import (
    SUPPORTED_ARCHITECTURES,
    PreTrainedModelWrapper,
    create_reference_model,
    unwrap_model_for_generation,
)
from . import AdaptiveKLController, BaseTrainer, FixedKLController, GRPOConfig, RunningMoments


if is_deepspeed_available():
    import deepspeed

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
library_name: transformers
tags:
- trl
- grpo
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/huggingface/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
"""

class GRPOTrainer(BaseTrainer):
    """
    The GRPOTrainer uses Gradient-based Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original DeepSeek Math work here:
    https://github.com/deepseek-ai and https://arxiv.org/pdf/2402.03300.pdf
    Note, this trainer is heavily inspired by the original PPO work in the ppo_trainer.py file here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py


    Attributes:
        **config** (`GRPOConfig`) -- Configuration object for GRPOTrainer. Check the documentation of `GRPOConfig` for more
            details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    _tag_names = ["trl", "grpo", "transformers", "reinforcement-learning"]

    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        model: Optional[PreTrainedModelWrapper] = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_data_collator: Optional[typing.Callable] = None,
    ):
        """
        Initialize GRPOTrainer.

        Args:
            config (`GRPOConfig`):
                Configuration object for GRPOTrainer. Check the documentation of `GRPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (`Optional[torch.optim.Optimizer]`):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function that is going to be used for `prepare_dataloader` method. Note this collator
                is different from the one we use for training. Pass a valid `training_data_collator` instead.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (`Optional[torch.optim.lr_scheduler]`):
                Learning rate scheduler used for training.
            training_data_collator (Optional[function]):
                Custom data collator used for training.
        """
        super().__init__(config)

        set_seed(config.seed)

         # Step 0: check positional arguments validity
        if not isinstance(config, GRPOConfig):
            raise ValueError(f"config must be a GRPOConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizerBase like a PreTrainedTokenizer or a PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        config.world_size = self.accelerator.num_processes
        config.global_backward_batch_size = config.backward_batch_size * config.world_size
        config.global_batch_size = config.batch_size * config.world_size

        self.model = model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_grpo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )
        self.is_using_text_environment = getattr(config, "use_text_environment", False)

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
            if num_shared_layers is not None:
                warnings.warn(
                    "num_shared_layers is ignored when ref_model is provided. Two different models are used for the "
                    "model and the reference model and no layers are shared.",
                    UserWarning,
                )
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None`, got {type(ref_model)} - supported "
                f"architectures are: {SUPPORTED_ARCHITECTURES} "
            )
        self.optional_peft_ctx = (
            self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
            if self.is_peft_model
            else nullcontext
        )

        if not (isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        if dataset is not None and not (isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, Dataset)):
            raise ValueError("dataset must be a torch.utils.data.Dataset or datasets.Dataset")
        elif dataset is None:
            warnings.warn(
                "No dataset is provided. Make sure to set config.batch_size to the correct value before training.",
                UserWarning,
            )
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                "No dataset is provided. In a multi-GPU setting, this will lead to an error. You should"
                " prepare your dataloader yourself with `dataloader = grpo_trainer.accelerator.prepare(dataloader)`"
                " and using `torch.utils.data.DataLoader`, or pass a dataset to the `GRPOTrainer`. Please "
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # Step 3: Initialize optimizer and data collator
        if training_data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        else:
            self.data_collator = training_data_collator
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )


        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                # For backward compatibility with older versions of transformers
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.model.pretrained_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        if is_deepspeed_used:
            # Quantized models are already set on the correct device
            if not self.is_peft_model and not (
                getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self.accelerator.prepare(self.ref_model)

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.num_processes > 1

        # init the current step
        self.current_step = 0

        # init variables for pushing model to hub
        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError("You have to specify repo_id in order to push the model to the hub!")
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

            if not getattr(self.model, "is_sequential_parallel", False):
                self.current_device = self.accelerator.device
            else:
                if is_xpu_available():
                    self.current_device = torch.device("xpu:0")
                elif is_npu_available():
                    self.current_device = torch.device("npu:0")
                else:
                    self.current_device = torch.device("cuda:0")

        GRPODecorators.optimize_device_cache = self.config.optimize_device_cache
        self.running = RunningMoments(self.accelerator)

        self.use_process_supervision = self.config.use_process_supervision
        self.use_iterative_reward_model_training = self.config.use_iterative_reward_model_training
        self.sampling_group_size = self.config.sampling_group_size

        self.sampling_strategy = self.config.sampling_strategy
        self.sampling_strategy_top_p = self.config.sampling_strategy_top_p
        self.sampling_strategy_top_k = self.config.sampling_strategy_top_k

        self.sampling_properties: dict = {"do_sample": True , "num_return_sequences": self.config.sampling_group_size}

        if self.sampling_strategy == "top_p":
            self.sampling_properties["top_p"] = self.sampling_strategy_top_p
            self.sampling_properties["top_k"] = 0

        elif self.sampling_strategy == "top_k":
            self.sampling_properties["top_k"] = self.sampling_strategy_top_k



    def _filter_kwargs(self, kwargs, target_func):
        """
        filter the keyword arguments that are supported by the target function.

        Args:
            kwargs (dict):
                Keyword arguments
            target_func (function):
                Target function
        """
        return {k: v for k, v in kwargs.items() if k in inspect.signature(target_func).parameters.keys()}



    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

        # Adapted from transformers.Trainer._remove_unused_columns
    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    # Adapted from transformers.Trainer._set_signature_columns_if_needed
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # query | we need query  for logging purpose
            self._signature_columns += [ "query"]

    def generate(
        self,
        query_tensor:  List[torch.Tensor],
        length_sampler: Optional[Callable] = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        generate_ref_response: bool = False,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
            generate_ref_response (`bool`, *optional*):
                If set to `True` the reference response is also generated, defaults to `False`.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """
        if generate_ref_response:
            ref_model = self.model if self.is_peft_model else self.ref_model

        if isinstance(query_tensor, List):
            response = self._generate_batched(
                self.model,
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
            if generate_ref_response:
                ref_response = self._generate_batched(
                    ref_model,
                    query_tensor,
                    length_sampler=length_sampler,
                    batch_size=batch_size,
                    return_prompt=return_prompt,
                    **generation_kwargs,
                )
        else:
            if len(query_tensor.shape) == 2:
                raise ValueError(
                    "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
                )

            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            response = generate_batched(self.model, [query_tensor])
            if generate_ref_response:
                ref_response = generate_batched(ref_model, [query_tensor])

            # Unwraping the response, as it is a list of tensors
            response = response[0]
            if generate_ref_response:
                ref_response = ref_response[0]

        if generate_ref_response:
            return response, ref_response
        return response

    def _generate_batched(
            self,
            model: PreTrainedModelWrapper,
            query_tensors: List[torch.Tensor],
            length_sampler: Optional[Callable] = None,
            batch_size: int = 4,
            return_prompt: bool = True,
            pad_to_multiple_of: Optional[int] = None,
            remove_padding: bool = True,
            **generation_kwargs,
        ):

        outputs = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # Update generation_kwargs for nucleus sampling/top-k sampling
        generation_kwargs.update(self.sampling_properties)

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            expanded_inputs = {
                        k: v.repeat_interleave(self.sampling_group_size, dim=0)
                        for k, v in padded_inputs.items()
                    }

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generations = unwrapped_model.generate(**expanded_inputs, **generation_kwargs)

            for idx in range(0, len(generations), self.sampling_group_size):
                sample_outputs = []
                for j in range(self.sampling_group_size):
                    generation = generations[idx + j]
                    mask = expanded_inputs["attention_mask"][idx + j]

                    if not self.is_encoder_decoder:
                        output = generation[(1 - mask).sum() :]  # remove padding
                    else:
                        output = generation
                    if not return_prompt and not self.is_encoder_decoder:
                        output = output[(mask).sum() :]  # remove prompt
                    if remove_padding and self.tokenizer.eos_token_id in output:
                        pad_mask = output == self.tokenizer.eos_token_id
                        pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                        output = output[: pad_start + 1]  # keep the eos token at the end
                    sample_outputs.append(output)
                outputs.append(sample_outputs)

            self.tokenizer.padding_side = padding_side_default
            return outputs
