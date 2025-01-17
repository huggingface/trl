# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import random
import warnings
from typing import Any, Callable, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from torch.utils.data import IterableDataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    EvalPrediction,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainingArguments,
    is_apex_available,
)
from transformers.training_args import OptimizerNames
from transformers.utils import logging

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..extras.dataset_formatting import get_formatting_func_from_dataset
from ..models.utils import unwrap_model_for_generation
from .base_online_trainer import BaseOnlineTrainer
from .judges import BasePairwiseJudge
from .utils import DPODataCollatorWithPadding, empty_cache, get_reward, truncate_right


if is_apex_available():
    from apex import amp

logger = logging.get_logger(__name__)


class RLOOv3Trainer(BaseOnlineTrainer):
    _tag_names = ["trl", "rloo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        ref_model: Union[PreTrainedModel, nn.Module, None] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        reward_processing_class: Optional[PreTrainedTokenizerBase] = None,
        peft_config: Optional[dict] = None,
        formatting_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if formatting_func is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays None
            formatting_func = get_formatting_func_from_dataset(train_dataset, processing_class)
            # if a template is detected, we don't need to add special tokens again
            if formatting_func is not None:
                args.dataset_kwargs["add_special_tokens"] = False

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    processing_class,
                    reward_processing_class,
                    args.dataset_text_field,
                    formatting_func,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **args.dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        processing_class,
                        reward_processing_class,
                        args.dataset_text_field,
                        formatting_func,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **args.dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_class=reward_processing_class,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.stats = {
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "objective/advantage_var": [],
            "val/contain_eos_token": [],
            "policy/approx_kl": [],
            "policy/clipfrac": [],
            "policy/ratio": [],
            "policy/ratio_var": [],
        }
        if self.reward_model is not None:
            self.stats["objective/rlhf_reward"] = []
            self.stats["objective/scores"] = []

        self.batch_num = 0
        self.batch_indices_to_sample = []
        self.prompt_completion_batches = []

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        reward_processing_class,
        dataset_text_field: str,
        formatting_func: Optional[Callable],
        remove_unused_columns=False,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        if skip_prepare_dataset:
            return dataset

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "valid formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            def formatting_func(x):
                return x["input_ids"]

        return self._prepare_dataloader(
            processing_class,
            reward_processing_class,
            dataset,
            dataset_text_field,
            formatting_func,
            add_special_tokens,
            remove_unused_columns,
        )

    def _prepare_dataloader(
        self,
        processing_class,
        reward_processing_class,
        dataset,
        dataset_text_field: str,
        formatting_func: Optional[Callable] = None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = processing_class(
                element[dataset_text_field] if formatting_func is None else formatting_func(element),
                add_special_tokens=add_special_tokens,
                padding=False,
                return_length=False,
            )

            if formatting_func is not None and not isinstance(formatting_func(element), list):
                raise ValueError(
                    "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                )

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        if dataset.column_names is not None:  # None for IterableDataset
            extra_columns = list(set(dataset.column_names) - set(signature_columns))
        else:
            extra_columns = []

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
                "the default collator and yield to errors. If you want to inspect dataset other columns (in this "
                f"case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the "
                "default collator and create your own data collator in order to inspect the unused dataset columns.",
                UserWarning,
            )

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names if remove_unused_columns else None,
            # "batch_size": self.dataset_batch_size,
        }
        # if isinstance(dataset, datasets.Dataset):
        #     map_kwargs["num_proc"] = self.dataset_num_proc  # this arg is not available for IterableDataset
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)

        if reward_processing_class is not None:
            raise NotImplementedError("TODO")

        return tokenized_dataset

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)

        # Decode the completions, and format them if the input is conversational
        if self.reward_processing_class is None:
            reward_prompt_completion_ids = inputs["prompt_completion_input_ids"]
            reward_pad_token_id = self.processing_class.pad_token_id
            context_length = inputs["context_length"]
        else:
            reward_pad_token_id = self.reward_processing_class.pad_token_id
            raise NotImplementedError("separate reward tokenizer todo")
            # device = prompt_completion_ids.device

            # completion_ids = prompt_completion_ids[:, context_length:]
            # completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            # if is_conversational({"prompt": prompts[0]}):
            # examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
            # examples = [apply_chat_template(example, self.reward_processing_class) for example in examples]
            # prompts = [example["prompt"] for example in examples]
            # completions = [example["completion"] for example in examples]

            # Tokenize the prompts

            # Tokenize the completions
            # reward_completions_ids = self.reward_processing_class(
            #     completions, padding=True, return_tensors="pt", padding_side="right"
            # )["input_ids"].to(device)

            # Concatenate the prompts and completions and get the reward
            # reward_prompt_completion_ids = torch.cat((reward_prompt_ids, reward_completions_ids), dim=1)

        with torch.inference_mode():
            _, scores, _ = get_reward(
                self.reward_model,
                reward_prompt_completion_ids,
                reward_pad_token_id,
                context_length,
            )

            # Filter completion. Ensure that the sample contains stop_token_id
            # Completions not passing that filter will receive a lower score.
            contain_eos_token = torch.any(inputs["completion_ids"] == self.processing_class.eos_token_id, dim=-1)
            if self.args.missing_eos_penalty is not None:
                scores[~contain_eos_token] -= self.args.missing_eos_penalty

        # Normalize rewards
        if self.args.normalize_reward:
            scores = (scores - scores.mean()) / (scores.std() + 1e-8)
            scores = torch.clamp(scores, -self.args.reward_clip_range, self.args.reward_clip_range)

        # Compute total reward with KL penalty
        kl = inputs["gen_logprobs"] - inputs["ref_logprobs"]
        if self.args.token_level_kl:
            # Token-level KL penalty: apply KL penalty per token
            token_kl_penalty = -self.args.kl_coef * kl
            non_score_reward = token_kl_penalty.sum(1)
        else:
            # Sequence-level KL penalty: sum KL across tokens first
            sequence_kl = kl.sum(1)
            non_score_reward = -self.args.kl_coef * sequence_kl
        rlhf_reward = scores + non_score_reward

        # vectorized RLOO advantages implementation
        rlhf_reward = rlhf_reward.reshape(self.args.rloo_k, -1)
        baseline = (rlhf_reward.sum(0) - rlhf_reward) / (self.args.rloo_k - 1)
        advantages = rlhf_reward - baseline
        advantages = advantages.flatten()

        # Normalize advantages
        if self.args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # RLOO loss with PPO-style importance sampling and clipping
        output = model(
            inputs["prompt_completion_input_ids"], attention_mask=inputs["prompt_completion_attention_mask"]
        )
        # Compute new logprobs
        logits = output.logits[:, context_length - 1 : -1]
        logits /= self.args.temperature + 1e-7
        # Turn logits into logprobs
        all_logprobs = F.log_softmax(logits, dim=-1)
        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(all_logprobs, inputs["completion_ids"].unsqueeze(-1), dim=2).squeeze(-1)

        # Compute probability ratios
        logprobs_diff = logprobs.sum(1) - inputs["gen_logprobs"].sum(1)
        ratio = torch.exp(logprobs_diff)

        # PPO-style clipped loss
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        loss = pg_loss_max.mean()

        # Log everything
        self.stats["objective/advantage_var"].append(
            self.accelerator.gather_for_metrics(advantages.var()).mean().item()
        )

        # RLOO / PPO stats
        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())

        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
        self.stats["policy/clipfrac"].append(self.accelerator.gather_for_metrics(pg_clipfrac).mean().item())

        self.stats["policy/ratio"].append(self.accelerator.gather_for_metrics(ratio.mean()).mean().item())
        self.stats["policy/ratio_var"].append(self.accelerator.gather_for_metrics(ratio.var()).mean().item())

        approxkl = 0.5 * (logprobs_diff**2).mean()
        self.stats["policy/approx_kl"].append(self.accelerator.gather_for_metrics(approxkl).mean().item())

        # Default on-policy stats
        if self.reward_model is not None:
            self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(scores.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(
            self.accelerator.gather_for_metrics(contain_eos_token.float().mean()).mean().item()
        )

        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(kl.sum(1).mean()).mean().item())
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(non_score_reward.mean()).mean().item()
        )
        if self.reward_model is not None:
            rlhf_reward = scores + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

    def get_batch_samples(self, epoch_iterator, num_batches):
        """Get num_mini_batches batches of prompt samples from dataloader

        Generate completions for each prompt and get logprobs of completions under the generating model

        Treat this as a dataset and return it num_ppo_epochs number of times"""

        # if we've just started or sampled these batch(es) enough times, get new prompts
        if (
            not self.prompt_completion_batches
            or self.batch_num > self.args.num_ppo_epochs * self.args.num_mini_batches
        ):
            prompt_batch_samples, _ = super().get_batch_samples(
                epoch_iterator, num_batches * self.args.num_mini_batches
            )
            self.batch_num = 0
            batch_indices_randomized = [
                np.random.permutation(len(prompt_batch_samples)) for _ in range(self.args.num_ppo_epochs)
            ]
            # should be of length num_batches * num_mini_batches * num_ppo_epochs
            self.batch_indices_to_sample = np.concatenate(batch_indices_randomized)

            # TODO we can add chunking of forward_batch_size here
            self.prompt_completion_batches = []
            for inputs in prompt_batch_samples:
                inputs = self._prepare_inputs(inputs)
                _, context_length = inputs["input_ids"].shape
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    output = unwrapped_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        generation_config=self.generation_config,
                    )

                completion_ids = output[:, context_length:]
                completion_ids, completion_mask = truncate_right(
                    completion_ids, self.processing_class.eos_token_id, self.processing_class.pad_token_id
                )
                prompt_completion_ids = torch.cat((inputs["input_ids"], completion_ids), dim=1)
                prompt_completion_mask = torch.cat((inputs["attention_mask"], completion_mask), dim=1)

                with torch.no_grad():
                    # Get the logprobs of the completions from the model
                    output = self.model(prompt_completion_ids, attention_mask=prompt_completion_mask)
                    # There is 1 offset, because the model predict the next token
                    logits = output.logits[:, context_length - 1 : -1]
                    # Turn logits into logprobs
                    all_logprobs = F.log_softmax(logits, dim=-1)
                    # Take the completion tokens logprob
                    logprobs = torch.take_along_dim(all_logprobs, completion_ids.unsqueeze(-1), dim=2).squeeze(-1)

                    # Get logprobs under reference model
                    if self.ref_model is not None:
                        ref_output = self.ref_model(
                            prompt_completion_ids,
                            attention_mask=prompt_completion_mask,
                        )
                    else:  # peft case: we just need to disable the adapter
                        with self.model.disable_adapter():
                            ref_output = self.model(prompt_completion_ids, attention_mask=prompt_completion_mask)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
                    ref_logprobs = torch.take_along_dim(ref_all_logprobs, completion_ids.unsqueeze(-1), dim=2).squeeze(
                        -1
                    )

                # masking out logprobs past the EOS token
                past_eos_mask = completion_mask == 0
                logprobs = torch.masked_fill(logprobs, past_eos_mask, 0.0)
                ref_logprobs = torch.masked_fill(ref_logprobs, past_eos_mask, 0.0)

                # None for num_tokens
                self.prompt_completion_batches.append(
                    {
                        "prompt_input_ids": inputs["input_ids"],
                        "prompt_attention_mask": inputs["attention_mask"],
                        "prompt_completion_input_ids": prompt_completion_ids,
                        "prompt_completion_attention_mask": prompt_completion_mask,
                        "completion_ids": completion_ids,
                        "completion_mask": completion_mask,
                        "context_length": context_length,
                        "gen_logprobs": logprobs,
                        "ref_logprobs": ref_logprobs,
                    },
                )

                del inputs, output, logits, all_logprobs, ref_output, ref_logits, ref_all_logprobs  # free memory

        indices_to_sample = self.batch_indices_to_sample[
            self.batch_num * num_batches : (self.batch_num + 1) * num_batches
        ]
        self.batch_num += 1
        return [self.prompt_completion_batches[index] for index in indices_to_sample], None
