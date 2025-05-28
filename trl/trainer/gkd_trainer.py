# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
import random
import textwrap
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..extras.profiling import profiling_context
from ..models import prepare_deepspeed
from ..models.utils import unwrap_model_for_generation
from .gkd_config import GKDConfig
from .sft_trainer import SFTTrainer
from .utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
)


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb


class GKDTrainer(SFTTrainer):
    _tag_names = ["trl", "gkd"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        teacher_model: Union[PreTrainedModel, nn.Module, str] = None,
        args: Optional[GKDConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
    ):
        # add remove_unused_columns=False to the dataclass args
        args.remove_unused_columns = False
        data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["torch_dtype"] = (
                teacher_model_init_kwargs["torch_dtype"]
                if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            # For FSDP, we need device_placement=True to ensure proper device placement
            self.teacher_model = self.accelerator.prepare_model(
                teacher_model, evaluation_mode=True, device_placement=True
            )

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=0,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=self.processing_class.pad_token_id,
        )
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Training arguments
        self.use_transformers_paged = args.use_transformers_paged

    def _prepare_dataset(self, dataset, *args):
        # SFTTrainer._prepare_dataset() applies the chat template and rename the messages column to text. However, we
        # need to keep the messages column as it is. We use the following workaround to keep the messages column.
        dataset = dataset.add_column("_messages", dataset["messages"])
        dataset = super()._prepare_dataset(dataset, *args)
        dataset = dataset.rename_column("_messages", "messages")
        return dataset

    @staticmethod
    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            labels: Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing loss
            beta: Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature: Softmax temperature (default: 1.0)
            reduction: Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute student output
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # compute teacher output in eval mode with proper FSDP handling
        self.teacher_model.eval()
        with torch.no_grad():
            # For FSDP, we need to properly handle the teacher model
            if self.is_fsdp_enabled:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                # Use FSDP.summon_full_params to ensure all parameters are available
                with FSDP.summon_full_params(self.teacher_model, recurse=False):
                    outputs_teacher = self.teacher_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
            else:
                # For non-FSDP models, ensure inputs are on the teacher model's device
                teacher_device = next(self.teacher_model.parameters()).device
                outputs_teacher = self.teacher_model(
                    input_ids=inputs["input_ids"].to(teacher_device),
                    attention_mask=inputs["attention_mask"].to(teacher_device),
                )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        # Labels are the completion tokens we want to predict
        shifted_labels = inputs["input_ids"][:, prompt_lengths:].to(shifted_student_logits.device)

        # compute the loss
        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
        )

        return (loss, outputs_student) if return_outputs else loss

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        # Import FSDP for both generation paths

        # Generate output with respect to the prompt only
        if self.use_transformers_paged:
            # Use paged attention for generation
            # Extract prompts from inputs - handle both tensor and text formats
            if "prompts" in inputs:
                if isinstance(inputs["prompts"], torch.Tensor):
                    # Convert tensor prompts to text
                    prompts_text = self.processing_class.batch_decode(inputs["prompts"], skip_special_tokens=True)
                else:
                    prompts_text = inputs["prompts"]
            else:
                # Fallback: assume input_ids contains the prompts
                prompts_text = self.processing_class.batch_decode(inputs["input_ids"], skip_special_tokens=True)

            prompt_inputs = self.processing_class(text=prompts_text)
            generation_config.max_batch_tokens = 512
            generation_config.num_blocks = 1024
            generation_config.block_size = 128
            generation_config.do_sample = False  # logit processing issue for now
            generation_config.max_new_tokens = generation_config.max_new_tokens or 128

            previous_attn = model.config._attn_implementation
            if torch.cuda.is_available():
                model.config._attn_implementation = "paged_attention"
            else:
                model.config._attn_implementation = "sdpa_paged"

            model.eval()
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(model, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                unwrapped_model.to(torch.bfloat16)
                all_outputs = unwrapped_model.generate_batch(
                    prompt_inputs.input_ids, generation_config=generation_config
                )
                completion_ids = [output.generated_tokens for output in all_outputs.values()]
                completion_ids = [torch.tensor(ids, device=unwrapped_model.device) for ids in completion_ids]
                completion_ids = pad(completion_ids, padding_value=pad_token_id)
                prompt_ids = [torch.tensor(ids, device=unwrapped_model.device) for ids in prompt_inputs.input_ids]
                prompt_ids = pad(prompt_ids, padding_value=pad_token_id)
                generated_tokens = torch.cat([prompt_ids, completion_ids], dim=1)

            model.config._attn_implementation = previous_attn
            model.train()
        else:
            # Regular generation path with proper FSDP handling
            model.eval()
            with (
                unwrap_model_for_generation(
                    model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(model, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                generated_outputs = unwrapped_model.generate(
                    input_ids=inputs["prompts"],
                    attention_mask=inputs.get("prompt_attention_mask", None),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )
                generated_tokens = generated_outputs.sequences
            model.train()

        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper.
        With probability `self.lmbda`, it generates new responses using the student model,
        which are then used for training instead of the original inputs.
        """
        if self.seq_kd:
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
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
        @inproceedings{agarwal2024on-policy,
            title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
            author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
            year         = 2024,
            booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
            publisher    = {OpenReview.net},
            url          = {https://openreview.net/forum?id=3zKtaqxLhW},
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GKD",
            trainer_citation=citation,
            paper_title="On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
            paper_id="2306.13649",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
