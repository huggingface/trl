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

import warnings
from typing import Any, Callable, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from datasets import Dataset
from packaging import version
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    EvalPrediction,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_apex_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available, logging

from ..data_utils import apply_chat_template, is_conversational
from ..extras.dataset_formatting import get_formatting_func_from_dataset
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .rloov2_config import RLOOv2Config
from .utils import empty_cache, get_reward, truncate_right


# from .utils import DPODataCollatorWithPadding

if is_peft_available():
    from peft import get_peft_model

if is_apex_available():
    from apex import amp

logger = logging.get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RLOOv2Trainer(Trainer):
    """
    Trainer for the Reinforce-Leave-One-Out method.

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "rloo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_model: Union[str, PreTrainedModel],
        args: Optional[RLOOv2Config] = None,
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
        # Models
        # Trained model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        self.reward_model = reward_model

        # Preprocess and format dataset
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
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
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
        if self.reward_funcs is not None:
            self.stats["objective/rlhf_reward"] = []
            self.stats["objective/scores"] = []

        self.batch_index = 0
        self.batch_indices_to_sample = []
        self.prompt_completion_batches = []

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=args.num_generations,
            pad_token_id=processing_class.pad_token_id,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

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

        return self._preprocess_dataset(
            processing_class,
            reward_processing_class,
            dataset,
            dataset_text_field,
            formatting_func,
            add_special_tokens,
            remove_unused_columns,
        )

    def _preprocess_dataset(
        self,
        processing_class,
        reward_processing_class,
        dataset,
        dataset_text_field: str,
        formatting_func: Optional[Callable] = None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        # is_conversational only works for a single example
        # so figure out if its conversational before calling a batched .map
        # if formatting_func is not None:
        is_convo = is_conversational(dataset[0])

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            if formatting_func is not None:
                texts = formatting_func(element)
            elif is_convo:
                texts = apply_chat_template(element, tokenizer=processing_class)["prompt"]
            else:
                texts = element[dataset_text_field]

            outputs = processing_class(
                texts,
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
            raise NotImplementedError("Doesn't yet support reward processing class")

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
        if not self.prompt_completion_batches or self.batch_index > len(self.prompt_completion_batches):
            del self.prompt_completion_batches
            self.batch_index = 0

            prompt_batch_samples, _ = super().get_batch_samples(
                epoch_iterator, num_batches * self.args.num_mini_batches
            )
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
                # TODO multiple generations per prompt
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

        indices_to_sample = self.batch_indices_to_sample[self.batch_index : self.batch_index + num_batches]
        self.batch_index += num_batches
        return [self.prompt_completion_batches[index] for index in indices_to_sample], None

    def get_train_dataloader(self) -> DataLoader:
        dataloader = super().get_train_dataloader()

        # if num_ppo_epochs > 1 then we are actually taking num_ppo_epochs * len(dataloader) steps in an epoch
        # monkeypatch the length variable to get this behaviour in the training and eval loops
        if self.args.num_ppo_epochs > 1:
            dataloader.__len__ = lambda: self.args.num_ppo_epochs * len(dataloader)

        return dataloader

    # Same as Trainer._maybe_log_save_evaluate but log our metrics
    # start_time defaults to None to allow compatibility with transformers<=4.46
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                logs[key] = sum(val) / len(val)
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
                self.log(logs, start_time)
            else:  # transformers<=4.46
                self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == "best":
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
