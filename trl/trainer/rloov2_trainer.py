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

from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from ..models.utils import unwrap_model_for_generation
from .base_online_trainer import BaseOnlineTrainer
from .judges import BasePairwiseJudge
from .utils import DPODataCollatorWithPadding, empty_cache, get_reward, truncate_right


if is_apex_available():
    from apex import amp

logger = logging.get_logger(__name__)

INVALID_LOGPROB = 1.0


class RLOOv2Trainer(BaseOnlineTrainer):
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
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=processing_class.pad_token_id)

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
        }
        if self.reward_model is not None:
            self.stats["objective/rlhf_reward"] = []
            self.stats["objective/scores"] = []

    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
        """Tokenize a single row from a DPO specific dataset."""
        batch = tokenizer(feature["prompt"])
        batch = {f"prompt_{key}": value for key, value in batch.items()}
        return batch

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        # Apply chat template and tokenize the input.
        # We do this on-the-fly to enable the use of reward models and policies with different tokenizers / chat templates.
        batch_size = len(next(iter(inputs.values())))
        prompts = inputs["prompt"]
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [self.tokenize_row(x, self.model.config.is_encoder_decoder, self.processing_class) for x in inputs]
        inputs = self.data_collator(inputs)

        # Sample 1 completion per prompt of size `max_new_tokens` from the model
        inputs = self._prepare_inputs(inputs)
        _, context_length = inputs["prompt_input_ids"].shape
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=inputs["prompt_input_ids"],
                attention_mask=inputs["prompt_attention_mask"],
                generation_config=self.generation_config,
            )

        completion_ids = output[:, context_length:]
        completion_ids, completion_mask = truncate_right(
            completion_ids, self.processing_class.eos_token_id, self.processing_class.pad_token_id
        )
        contain_eos_token = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)
        prompt_completion_ids = torch.cat((inputs["prompt_input_ids"], completion_ids), dim=1)
        prompt_completion_mask = torch.cat((inputs["prompt_attention_mask"], completion_mask), dim=1)

        del inputs

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

        # compute KL divergence, masking out logprobs past the EOS token
        past_eos_mask = completion_mask == 0
        logprobs = torch.masked_fill(logprobs, past_eos_mask, 0.0)
        ref_logprobs = torch.masked_fill(ref_logprobs, past_eos_mask, 0.0)

        kl = logprobs - ref_logprobs

        # Decode the completions, and format them if the input is conversational
        device = prompt_completion_ids.device
        completions_ids = prompt_completion_ids[:, context_length:]
        completions = self.processing_class.batch_decode(completions_ids, skip_special_tokens=True)

        if is_conversational({"prompt": prompts[0]}):
            examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
            examples = [apply_chat_template(example, self.reward_processing_class) for example in examples]
            prompts = [example["prompt"] for example in examples]
            completions = [example["completion"] for example in examples]

        # Tokenize the prompts
        prompts_ids = self.reward_processing_class(prompts, padding=True, return_tensors="pt", padding_side="left")[
            "input_ids"
        ].to(device)
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

        # Normalize rewards
        if self.args.normalize_reward:
            scores = (scores - scores.mean()) / (scores.std() + 1e-8)
            scores = torch.clamp(scores, -self.args.reward_clip_range, self.args.reward_clip_range)

        # Compute total reward with KL penalty
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

        # REINFORCE loss
        # current implementation is fully on-policy and therefore doesn't require importance sampling or clipping.
        # future versions that implement off-policy learning should add this
        losses = -advantages * logprobs.sum(1)
        loss = losses.mean()

        # Log everything
        self.stats["objective/advantage_var"].append(
            self.accelerator.gather_for_metrics(advantages.var()).mean().item()
        )

        if self.reward_model is not None:
            self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(scores.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())

        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(kl.sum(1).mean()).mean().item())
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(non_score_reward.mean()).mean().item()
        )
        if self.reward_model is not None:
            rlhf_reward = scores + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())
        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())

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
