# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
import textwrap
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import jinja2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from datasets import Dataset
from packaging import version
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_apex_available,
    is_wandb_available,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled, logging

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .judges import BasePairwiseJudge
from .online_dpo_config import OnlineDPOConfig
from .utils import (
    SIMPLE_CHAT_TEMPLATE,
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    empty_cache,
    generate_model_card,
    get_reward,
    prepare_deepspeed,
    truncate_right,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model

if is_apex_available():
    from apex import amp


if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)


class OnlineDPOTrainer(Trainer):
    r"""
    Initialize OnlineDPOTrainer.

    Args:
        model (`transformers.PreTrainedModel` or `torch.nn.Module`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`transformers.PreTrainedModel` or `torch.nn.Module` or `None`):
            The reference model to use for training. If None is specified, the reference model will be created from
            the model.
        reward_model (`transformers.PreTrainedModel` or `torch.nn.Module` or `None`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        judge (`BasePairwiseJudge`):
            The judge to use for pairwise comparison of model completions.
        args (`OnlineDPOConfig`):
            The online DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        peft_config (`Dict`):
            The peft config to use for training.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "online-dpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        ref_model: Union[PreTrainedModel, nn.Module, None] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[OnlineDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        reward_processing_class: Optional[PreTrainedTokenizerBase] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, either omit the `ref_model` argument or pass `None`."
            )

        self.ref_model = ref_model

        if reward_model is not None and judge is not None:
            warnings.warn(
                "Both `reward_model` and `judge` are provided. Please choose provide only one of them. "
                "Ignoring `judge` and using `reward_model`."
            )
            judge = None
        elif reward_model is None and judge is None:
            raise ValueError("Either `reward_model` or `judge` must be provided.")

        self.reward_model = reward_model
        self.reward_processing_class = reward_processing_class
        self.judge = judge

        if args.missing_eos_penalty is not None and judge is not None:
            raise ValueError("`missing_eos_penalty` is not supported when `judge` is provided.")

        if args is None:
            raise ValueError("`args` must be provided.")

        # Check that the processing_class is provided
        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")

        # Convert to PEFT model if peft_config is provided
        if peft_config is not None:
            # Check if PEFT is available
            if not is_peft_available():
                raise ImportError(
                    "PEFT is not available and passed `peft_config`. Please install PEFT with "
                    "`pip install peft` to use it."
                )

            # If the model is already a PeftModel, we need to merge and unload it.
            # Further information here: https://huggingface.co/docs/trl/dpo_trainer#reference-model-considerations-with-peft
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            # Get peft model with the given config
            model = get_peft_model(model, peft_config)

        # Disable dropout in the model if specified
        if args.disable_dropout:
            disable_dropout_in_model(model)

        # Handle the ref_model
        # Usually, the user wants the ref model to be the initial version of the model. When using PEFT, it's easy to
        # get the ref model, as it's just the model with a disabled adapter. When not using PEFT, we need to create
        # the ref model from the model by copying it and disable the gradients and set it in evaluation mode.
        if ref_model is None:  # No ref model provided, the most common case
            if peft_config is None:
                self.ref_model = create_reference_model(model)  # copy, disable gradients, set eval mode
            else:
                self.ref_model = None  # we don't need a ref model here, we can just disable the adapter.
        else:  # rare case, the user provided a ref model
            self.ref_model = ref_model
            self.ref_model.eval()

        # Disable the gradient and set the reward model in eval mode
        if self.reward_model is not None:
            self.reward_model.eval()

        # Define the collator is not provided
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=processing_class.pad_token_id)

        self.stats = {
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "val/contain_eos_token": [],
            "beta": [],
        }
        if self.reward_model is not None:
            self.stats["objective/rlhf_reward"] = []
            self.stats["objective/scores_margin"] = []
            self.stats["objective/scores"] = []

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            use_cache=False if args.gradient_checkpointing else True,
        )

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

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._beta = args.beta

        # Placed after the super().__init__ because we need self.is_deepspeed_enabled and self.accelerator
        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.ref_model = prepare_deepspeed(self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16)
        else:
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)

    @property
    def beta(self):
        if isinstance(self._beta, list):
            epoch = self.state.epoch
            return self._beta[epoch] if epoch < len(self._beta) else self._beta[-1]
        else:
            return self._beta

    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
        """Tokenize a single row from a DPO specific dataset."""
        if not is_encoder_decoder:
            batch = tokenizer(feature["prompt"], add_special_tokens=False)
            # Add BOS token to head of prompt. Avoid adding if it's already there
            if tokenizer.bos_token_id is not None:
                prompt_len_input_ids = len(batch["input_ids"])
                if prompt_len_input_ids == 0 or tokenizer.bos_token_id != batch["input_ids"][0]:
                    batch["input_ids"] = [tokenizer.bos_token_id] + batch["input_ids"]
                    batch["attention_mask"] = [1] + batch["attention_mask"]
        else:
            batch = tokenizer(feature["prompt"], add_special_tokens=True)
        batch = {f"prompt_{key}": value for key, value in batch.items()}
        return batch

    # Same as Trainer.get_train_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # Same as Trainer.get_eval_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        model.train()

        # Apply chat template and tokenize the input.
        # We do this on-the-fly to enable the use of reward models and policies with different tokenizers / chat templates.
        batch_size = len(next(iter(inputs.values())))
        prompts = inputs["prompt"]
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [self.tokenize_row(x, self.model.config.is_encoder_decoder, self.processing_class) for x in inputs]
        inputs = self.data_collator(inputs)

        # Sample 2 completions per prompt of size `max_new_tokens` from the model
        inputs = self._prepare_inputs(inputs)
        num_examples, context_length = inputs["prompt_input_ids"].shape
        prompt_ids = inputs["prompt_input_ids"].repeat(2, 1)
        prompt_mask = inputs["prompt_attention_mask"].repeat(2, 1)
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )
        del inputs

        completion_ids = output[:, context_length:]
        completion_ids, completion_mask = truncate_right(
            completion_ids, self.processing_class.eos_token_id, self.processing_class.pad_token_id
        )
        contain_eos_token = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)
        # There is 1 offset, because the model predict the next token
        logits = output.logits[:, context_length - 1 : -1]
        # Turn logits into logprobs
        all_logprobs = F.log_softmax(logits, dim=-1)
        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(all_logprobs, completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        del output, logits, all_logprobs  # free memory

        # Same for the reference model
        with torch.no_grad():
            if self.ref_model is not None:
                ref_output = self.ref_model(prompt_completion_ids, attention_mask=prompt_completion_mask)
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    ref_output = self.model(prompt_completion_ids, attention_mask=prompt_completion_mask)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.take_along_dim(ref_all_logprobs, completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
            del ref_output, ref_logits, ref_all_logprobs  # free memory

        # Decode the completions, and format them if the input is conversational
        device = prompt_completion_ids.device
        completions_ids = prompt_completion_ids[:, context_length:]
        completions = self.processing_class.batch_decode(completions_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Get the reward from the reward model or judge
        if self.judge is not None:
            # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
            # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
            # independent of the model's chat template, we use the raw conversation data, and apply our own chat
            # template to it.
            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                prompts = [template.render(messages=prompt) for prompt in prompts]
                completions = [template.render(messages=completion) for completion in completions]

            ranks_of_first_completion = self.judge.judge(
                prompts, list(zip(completions[:num_examples], completions[num_examples:]))
            )

            # convert ranks to a True/False mask:
            # when rank == 0, it means the first completion is the best
            # when rank == 1, it means the second completion is the best
            mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)
        else:
            # The reward model may not have the same chat template or tokenizer as the model, so we need to use the
            # raw data (string), apply the chat template (if needed), and tokenize it with the reward processing class.
            prompts = 2 * prompts  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
            if is_conversational({"prompt": prompts[0]}):
                examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
                examples = [apply_chat_template(example, self.reward_processing_class) for example in examples]
                prompts = [example["prompt"] for example in examples]
                completions = [example["completion"] for example in examples]

            # Tokenize the prompts
            prompts_ids = self.reward_processing_class(
                prompts, padding=True, return_tensors="pt", padding_side="left"
            )["input_ids"].to(device)
            context_length = prompts_ids.shape[1]

            # Tokenize the completions
            completions_ids = self.reward_processing_class(
                completions, padding=True, return_tensors="pt", padding_side="right"
            )["input_ids"].to(device)

            # Concatenate the prompts and completions and get the reward
            prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
            with torch.inference_mode():
                _, scores, _ = get_reward(
                    self.reward_model, prompt_completion_ids, self.reward_processing_class.pad_token_id, context_length
                )

                # Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

            # Split the scores in 2 (the prompts of the first half are the same as the second half)
            first_half, second_half = scores.split(num_examples)

            # Get the indices of the chosen and rejected examples
            mask = first_half >= second_half

        num_examples_range = torch.arange(num_examples, device=device)
        chosen_indices = num_examples_range + (~mask * num_examples)
        rejected_indices = num_examples_range + (mask * num_examples)

        # Build tensor so that the first half is the chosen examples and the second half the rejected examples
        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        # Split the chosen and rejected examples
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, num_examples)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, num_examples)
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.loss_type}")

        loss = losses.mean()

        # Log everything
        if self.reward_model is not None:
            scores_margin = scores[chosen_indices] - scores[rejected_indices]
            self.stats["objective/scores_margin"].append(self.accelerator.gather(scores_margin.mean()).mean().item())
            self.stats["objective/scores"].append(self.accelerator.gather(scores.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        self.stats["logps/chosen"].append(self.accelerator.gather(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather(rejected_logprobs_sum).mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather(mean_kl).mean().item())
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(self.accelerator.gather(mean_non_score_reward).mean().item())
        if self.reward_model is not None:
            rlhf_reward = scores + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather(rlhf_reward).mean().item())
        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather(mean_entropy).mean().item())
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
        self.stats["beta"].append(self.beta)

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

    # Same as Trainer.evaluate but log our metrics
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}

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

    # Copy-pasted from transformers.Trainer to maintain compatibility with earlier versions.
    # This can be removed once the minimum transformers version is updated to 4.47.
    # Refer to https://github.com/huggingface/trl/pull/2288 for more details.
    def _determine_best_metric(self, metrics, trial):
        """
        Determine if the model should be saved based on the evaluation metrics.
        If args.metric_for_best_model is not set, the loss is used.
        Returns:
            bool: True if a new best metric was found, else False
        """
        is_new_best_metric = False

        if self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model

            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less

            if self.state.best_metric is None:
                self.state.best_metric = float("-inf") if self.args.greater_is_better else float("inf")

            if operator(metric_value, self.state.best_metric):
                run_dir = self._get_output_dir(trial=trial)
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                output_dir = os.path.join(run_dir, checkpoint_folder)
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                is_new_best_metric = True

        return is_new_best_metric

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `List[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{guo2024direct,
            title        = {{Direct Language Model Alignment from Online AI Feedback}},
            author       = {Shangmin Guo and Biao Zhang and Tianlin Liu and Tianqi Liu and Misha Khalman and Felipe Llinares and Alexandre Ram{\'{e}} and Thomas Mesnard and Yao Zhao and Bilal Piot and Johan Ferret and Mathieu Blondel},
            year         = 2024,
            eprint       = {arXiv:2402.04792}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="Online DPO",
            trainer_citation=citation,
            paper_title="Direct Language Model Alignment from Online AI Feedback",
            paper_id="2402.04792",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))
