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
from accelerate.utils import broadcast_object_list, gather_object
from datasets import Dataset
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

from ..extras.vllm_client import VLLMClient
from ..import_utils import is_vllm_available
from ..models import prepare_deepspeed
from ..models.utils import unwrap_model_for_generation
from .gkd_config import GKDConfig
from .sft_trainer import SFTTrainer
from .utils import (
    DataCollatorForChatML,
    disable_dropout_in_model,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    pad,
)
from ..data_utils import maybe_apply_chat_template


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


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
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path

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
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

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

        # vLLM setup for student model if enabled
        self.student_use_vllm = args.student_use_vllm
        if self.student_use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and student_use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.student_vllm_mode = args.student_vllm_mode
            if self.student_vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.student_vllm_client = VLLMClient(
                        args.student_vllm_server_host,
                        args.student_vllm_server_port,
                        connection_timeout=args.student_vllm_server_timeout,
                    )
                    self.student_vllm_client.init_communicator()
            elif self.student_vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path
                self.student_llm = LLM(
                    model=student_model_name_or_path,
                    tensor_parallel_size=args.student_vllm_tensor_parallel_size,
                    gpu_memory_utilization=args.student_vllm_gpu_memory_utilization,
                    # Max num seqs can be a small number as we generate one by one during training
                    max_num_seqs=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,  # Assuming max_length covers prompt + new tokens
                    seed=args.seed,  # Use the global seed for consistency
                )
            else:
                raise ValueError(f"Unknown student_vllm_mode: {self.student_vllm_mode}")
            self.student_vllm_guided_decoding_regex = args.student_vllm_guided_decoding_regex
            self.student_vllm_sync_frequency = args.student_vllm_sync_frequency
            self._last_student_sync_step = -1

    def _prepare_dataset(self, dataset, *args):
        # SFTTrainer._prepare_dataset() applies the chat template and rename the messages column to text. However, we
        # need to keep the messages column as it is. We use the following workaround to keep the messages column.
        # Only do this if a "prompt" column doesn't already exist from user script preprocessing
        if "prompt" not in dataset.column_names:
            if "messages" in dataset.column_names: # Check if "messages" column exists before trying to access it
                dataset = dataset.add_column("_messages", dataset["messages"])
                dataset = super()._prepare_dataset(dataset, *args)
                dataset = dataset.rename_column("_messages", "messages")
            else: # If "messages" is not there (e.g. user provided text/completion), just call super
                dataset = super()._prepare_dataset(dataset, *args)
        else: # If "prompt" column exists, assume user has preprocessed, just call super
            dataset = super()._prepare_dataset(dataset, *args)
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
                torch.stack([student_log_probs + torch.log1p(- beta), teacher_log_probs + torch.log(beta)]),
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

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]

        # compute loss
        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
        )

        # empty cache
        empty_cache()

        # Return loss
        return (loss, outputs_student) if return_outputs else loss

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        # Generate output with respect to the prompt only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs.get("prompt_attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def _generate_on_policy_outputs_student_vllm(self, inputs, generation_config, pad_token_id=None):
        device = self.accelerator.device
        # Decode the tokenized prompts from inputs["prompts"]
        # Ensure to skip special tokens and padding tokens during decoding
        prompts_text = self.processing_class.batch_decode(
            inputs["prompts"],
            skip_special_tokens=True,
            # clean_up_tokenization_spaces=False # Keep this commented unless specific issues arise
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text = [p.replace(self.processing_class.pad_token, "") for p in prompts_text]

        max_new_tokens = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0

        if self.student_vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                completion_ids = self.student_vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_new_tokens,
                    guided_decoding_regex=self.student_vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]
        elif self.student_vllm_mode == "colocate":
            if self.student_vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.student_vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_new_tokens,
                guided_decoding=guided_decoding,
            )
            all_outputs = self.student_llm.generate(prompts_text, sampling_params=sampling_params, use_tqdm=False)
            completion_ids = [output.token_ids for outputs_list in all_outputs for output in outputs_list.outputs]
        else:
            raise ValueError(f"Unknown student_vllm_mode: {self.student_vllm_mode}")

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        prompt_tokenized = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.args.max_length - max_new_tokens,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_new_tokens length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_new_tokens:
                # Truncate if longer than max_new_tokens
                padded_completion_ids_list.append(completion_tensor[:max_new_tokens])
            elif len(completion_tensor) < max_new_tokens:
                # Pad if shorter than max_new_tokens
                padding_needed = max_new_tokens - len(completion_tensor)
                padded_tensor = torch.cat([
                    completion_tensor, 
                    torch.full((padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype)
                ])
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)
        
        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Mask prompt tokens in labels
        prompt_lengths = prompt_ids.shape[1]
        new_labels[:, :prompt_lengths] = -100

        return new_input_ids, new_attention_mask, new_labels

    def _sync_fsdp_params_to_student_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_student_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if hasattr(module, '__class__') and 'FSDP' in module.__class__.__name__:
            # Import FSDP here to avoid import errors if not available
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                if isinstance(module, FSDP):
                    with FSDP.summon_full_params(module, recurse=False, writeback=False):
                        for param_name, param in module.named_parameters():
                            full_name = f"{prefix}.{param_name}" if prefix else param_name
                            for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                                full_name = full_name.replace(extra, "")

                            if full_name in visited:
                                continue  # skip FSDP subtrees already traversed
                            visited.add(full_name)

                            if self.student_vllm_mode == "server" and self.accelerator.is_main_process:
                                self.student_vllm_client.update_named_param(full_name, param.data)
                            elif self.student_vllm_mode == "colocate":
                                llm_model = self.student_llm.llm_engine.model_executor.driver_worker.model_runner.model
                                llm_model.load_weights([(full_name, param.data)])
            except ImportError:
                pass  # FSDP not available

    def _move_student_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            try:
                import deepspeed
                gather_if_zero3 = deepspeed.zero.GatheredParameters
            except ImportError:
                gather_if_zero3 = nullcontext
        else:
            gather_if_zero3 = nullcontext

        # Check if model is PEFT
        try:
            from peft import PeftModel
            is_peft_model_func = lambda model: isinstance(model, PeftModel)
        except ImportError:
            is_peft_model_func = lambda model: False

        if is_peft_model_func(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_student_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if hasattr(self.model, 'prefix') and self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.student_vllm_mode == "server" and self.accelerator.is_main_process:
                            self.student_vllm_client.update_named_param(name, param.data)
                        elif self.student_vllm_mode == "colocate":
                            llm_model = self.student_llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_student_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.student_vllm_mode == "server" and self.accelerator.is_main_process:
                            self.student_vllm_client.update_named_param(name, param.data)
                        elif self.student_vllm_mode == "colocate":
                            llm_model = self.student_llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.student_vllm_mode == "server" and self.accelerator.is_main_process:
            self.student_vllm_client.reset_prefix_cache()
        elif self.student_vllm_mode == "colocate":
            self.student_llm.reset_prefix_cache()

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper.
        With probability `self.lmbda`, it generates new responses using the student model,
        which are then used for training instead of the original inputs.
        """
        if random.random() <= self.lmbda:
            if self.student_use_vllm:
                # # Synchronize student model weights to vLLM if needed
                # if (self.state.global_step % self.student_vllm_sync_frequency == 0 and 
                #     self.state.global_step != self._last_student_sync_step):
                #     self._move_student_model_to_vllm()
                #     self._last_student_sync_step = self.state.global_step

                new_input_ids, new_attention_mask, new_labels = self._generate_on_policy_outputs_student_vllm(
                    inputs, self.generation_config, self.processing_class.pad_token_id
                )
            else:
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
