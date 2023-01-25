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
import logging
import os
import random
import time
import warnings
from typing import List, Optional, Union

import datasets
import torch
from accelerate import Accelerator
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..core import (
    WANDB_PADDING,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    stack_dicts,
    stats_to_np,
    whiten,
)
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/lvwerra/trl) that has been fine-tuned with reinforcement learning to guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

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


class PPOTrainer(BaseTrainer):
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face transformer model with a casual language modelling head.
            Check the documentation of `PreTrainedModelWrapper` for more details. If no reference model is provided, the
            trainer will create a reference model with the same architecture as the model to be optimized with shared layers.
        **tokenizer** (`Union[PreTrainedTokenizer, PreTrainedTokenizerFast]`) -- Tokenizer to be used for encoding the data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided,
            the dataloader must be created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is provided, the trainer will create an Adam optimizer with
            the learning rate specified in the configuration object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference model, if no reference model is passed. If no number is provided, all the layers
            will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(
        self,
        config: PPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: PreTrainedModelWrapper = None,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator=None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize PPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizer`):
                Hugging Face tokenizer
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
        """
        super().__init__(config)
        # Step 0: check positional arguments validity
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizer or PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        if not isinstance(model, PreTrainedModelWrapper):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(log_with=config.log_with, **config.accelerator_kwargs)
        self.accelerator.init_trackers(config.tracker_project_name, config=config.to_dict(), **config.tracker_kwargs)

        self.model = model
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")

        if isinstance(ref_model, PreTrainedModelWrapper):
            self.ref_model = ref_model
            if num_shared_layers is not None:
                warnings.warn(
                    "num_shared_layers is ignored when ref_model is provided. Two different models are used for the model and the reference model and no layers are shared.",
                    UserWarning,
                )
        elif ref_model is None:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None`, got {type(ref_model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )

        if not (isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        if dataset is not None and not (isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, Dataset)):
            raise ValueError("dataloader must be a torch.utils.data.Dataset or datasets.Dataset")
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
                "No dataset is provided. In a multi-GPU setting, this will lead to an error. You should",
                " prepare your dataloader yourself with `dataloader = ppo_trainer.accelerator.prepare(dataloader)`",
                " and using `torch.utils.data.DataLoader`, or pass a dataset to the `PPOTrainer`. Please ",
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                raise ValueError("lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler")

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        (
            self.model,
            self.ref_model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.ref_model, self.optimizer, self.data_collator, self.dataloader, self.lr_scheduler
        )

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # init the current step
        self.current_step = 0

        # init wandb on the main process:
        if self.accelerator.is_main_process and self.config.log_with == "wandb":
            import wandb

            wandb.watch(self.model, log="all")

        if self.config.forward_batch_size > 1 and (self.is_encoder_decoder or self.tokenizer.padding_side == "left"):
            # warn users that this is not well supported yet
            logging.warning(
                "Forward batch size > 1 is not well supported yet for encoder-decoder models and when using `tokenizer.padding_side='left'`. This can lead to unexpected behaviour."
                " therefore, we recommend using forward_batch_size=1."
            )

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
        )
        return dataloader

    # Adapted from transformers.Trainer._set_signature_columns_if_needed
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += list(set(["label", "query", "response"]))

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
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def generate(self, query_tensor: torch.Tensor, **generation_kwargs):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`batch_size`, `seq_len`) containing query tokens.
            gen_kwargs (dict[str, Any]):
                Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """
        response = self.accelerator.unwrap_model(self.model).generate(
            query_tensor.unsqueeze(dim=0), **generation_kwargs
        )

        return response

    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(f"Elements in {name} must tensors - got {type(tensor_list[0])}")
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.accelerator.device) for tensor in queries]
        responses = [tensor.to(self.accelerator.device) for tensor in responses]
        scores = [tensor.to(self.accelerator.device) for tensor in scores]

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores

    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores = self._step_safety_checker(bs, queries, responses, scores)

        timing = dict()
        t0 = time.time()

        t = time.time()

        logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses)
        timing["time/ppo/forward_pass"] = time.time() - t

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing["time/ppo/compute_rewards"] = time.time() - t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.config.ppo_epochs):
            random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                train_stats = self.train_minibatch(
                    logprobs[idx].unsqueeze(0),
                    values[idx].unsqueeze(0),
                    rewards[idx].unsqueeze(0),
                    queries[idx].unsqueeze(0),
                    responses[idx].unsqueeze(0),
                    torch.cat([queries[idx], responses[idx]]).unsqueeze(0),
                )
                all_stats.append(train_stats)
        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(stats["objective/kl"], self.config.batch_size * self.accelerator.num_processes)

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def gather_stats(self, stats):
        """
        Gather stats from all processes. Useful in the context of distributed training.

        Args:
            stats (dict[str, Any]):
            a dictionary of stats to be gathered. The stats should contain torch tensors.

        Returns:
            `dict[str, Any]`: A dictionary of stats with the tensors gathered.
        """
        import torch.distributed as dist

        # Wait for all processes to finish
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                dist.all_reduce(v, dist.ReduceOp.SUM)
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    def batched_forward_pass(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses, shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses, shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = self.config.batch_size
        fbs = self.config.forward_batch_size
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        for i in range(int(bs / fbs)):
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]

            if self.is_encoder_decoder:
                input_ids = self.data_collator(query_batch)["input_ids"]
                decoder_input_ids = self.data_collator(response_batch)["input_ids"]

                input_kwargs = {
                    "input_ids": input_ids,
                    "decoder_input_ids": decoder_input_ids,
                }
            else:
                input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])[
                    "input_ids"
                ]

                input_kwargs = {
                    "input_ids": input_ids,
                }

            with torch.no_grad():
                logits, _, v = self.model(**input_kwargs)
                ref_logits, _, _ = self.ref_model(**input_kwargs)

            if self.is_encoder_decoder:
                logprobs = logprobs_from_logits(logits[:, :-1, :], decoder_input_ids[:, 1:])
                ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], decoder_input_ids[:, 1:])
            else:
                logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])

            for j in range(fbs):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = response_batch[j].shape[-1] - 1
                else:
                    start = len(query_batch[j]) - 1
                    end = len(query_batch[j]) + len(response_batch[j]) - 1

                if len(logprobs[j, start:end]) < 2:
                    raise ValueError("Responses are too short. Make sure they are at least 4 tokens long.")

                all_values.append(v[j, start - 1 : end - 1])
                all_logprobs.append(logprobs[j, start:end])
                all_ref_logprobs.append(ref_logprobs[j, start:end])

        return all_logprobs, all_ref_logprobs, all_values

    def train_minibatch(
        self,
        logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        query: torch.LongTensor,
        response: torch.LongTensor,
        model_input: torch.LongTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [batch_size, response_length]
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape [batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        loss_p, loss_v, train_stats = self.loss(logprobs, values, rewards, query, response, model_input)
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        t = time.time()
        self.optimizer.step()
        train_stats["time/ppo/optimizer_step"] = torch.Tensor([time.time() - t]).to(self.accelerator.device)
        return train_stats

    def compute_rewards(self, scores: torch.FloatTensor, logprobs: torch.FloatTensor, ref_logprobs: torch.FloatTensor):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            reward[-1] += score
            rewards.append(reward)
        return rewards, non_score_rewards

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        query: torch.LongTensor,
        response: torch.LongTensor,
        model_input: torch.LongTensor,
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `hidden_dim`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`)
            query (`torch.LongTensor`):
                Encoded queries, shape (`batch_size`, `query_length`)
            response (`torch.LongTensor`):
                Encoded responses, shape (`batch_size`, `response_length`)
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape (`batch_size`, `query_length+response_length`)
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        input_kwargs = {
            "input_ids": model_input,
        }

        if self.is_encoder_decoder:
            input_kwargs["input_ids"] = query
            input_kwargs["decoder_input_ids"] = response
            model_input = response

        logits, _, vpred = self.model(**input_kwargs)

        if self.is_encoder_decoder:
            logprob = logprobs_from_logits(logits[:, :-1, :], model_input[:, 1:])
            start, end = 1, response.shape[-1] - 1
            vpred = vpred[:, start - 1 : end - 1]
            logprob = logprob[:, start:end]
        else:
            logprob = logprobs_from_logits(logits[:, :-1, :], model_input[:, 1:])
            logprob, vpred = logprob[:, -gen_len:], vpred[:, -gen_len - 1 : -1]

        vpredclipped = clip_by_value(vpred, values - self.config.cliprange_value, values + self.config.cliprange_value)

        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.config.vf_coef * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = 0.5 * torch.mean((logprob - old_logprobs) ** 2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(
                entropy=entropy,
                approxkl=approxkl,
                policykl=policykl,
                clipfrac=pg_clipfrac,
                advantages=advantages,
                advantages_mean=torch.mean(advantages),
                ratio=ratio,
            ),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=torch.mean(vpred),
                error=torch.mean((vpred - returns) ** 2),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef: float, **data):
        """
        Record training step statistics.


        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
        kl_list = [logprobs - ref_logprobs for logprobs, ref_logprobs in zip(data["logprobs"], data["ref_logprobs"])]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data["logprobs"]]))
        mean_non_score_reward = torch.mean(
            torch.stack([torch.sum(non_score_reward) for non_score_reward in data["non_score_reward"]])
        )
        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
        }

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        return stats

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this containes the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """
        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}

            # Log stats
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards).to(self.accelerator.device)

            if "query" not in batch.keys() and "response" not in batch.keys():
                # warn the user that the game logs will not be logged
                warnings.warn(
                    "The game logs will not be logged because the batch does not contain the keys 'query' and 'response'."
                )
            elif self.config.log_with == "wandb":
                import wandb

                table_rows = [list(r) for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())]
                logs.update({"game_log": wandb.Table(columns=["query", "response", "reward"], rows=table_rows)})
            # All reduce rewards if distributed
            if self.is_distributed:
                import torch.distributed as dist

                dist.barrier()

                dist.all_reduce(rewards, op=torch.distributed.ReduceOp.SUM)
                rewards /= self.accelerator.num_processes

            logs.update(stats)
            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            if self.config.log_with == "tensorboard":
                # update the current step
                self.current_step += 1

            self.accelerator.log(logs, step=self.current_step if self.config.log_with == "tensorboard" else None)

        else:
            if self.is_distributed:
                import torch.distributed as dist

                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards).to(self.accelerator.device)

                dist.barrier()
                dist.all_reduce(rewards, op=torch.distributed.ReduceOp.SUM)

    def create_model_card(self, path: str, model_name: Optional[str] = "TRL Model") -> None:
        """Creates and saves a model card for a TRL model.

        Args:
            path (`str`): The path to save the model card to.
            model_name (`str`, *optional*): The name of the model, defaults to `TRL Model`.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        user = whoami()["name"]
        model_card_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, model_id=f"{user}/{path}")
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)

    def _save_pretrained(self, save_directory: str) -> None:
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.create_model_card(save_directory)
