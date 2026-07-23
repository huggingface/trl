# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import copy
import inspect
import random
import textwrap
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from accelerate.utils import gather_object, set_seed
from datasets import Dataset, IterableDataset
from packaging.version import Version
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, TrainerCallback, is_trackio_available, is_wandb_available
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_liger_kernel_available, is_peft_available, is_rich_available

from ...data_utils import is_conversational
from ...distributed import DistributedBackend
from ...extras.profiling import profiling_context, profiling_decorator
from ...generation.vllm_generation import VLLMGeneration
from ...import_utils import is_vllm_available
from ...models import prepare_deepspeed
from ...models.utils import _ForwardRedirection, unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    identity,
    pad,
    repeat_iterable_dataset,
    shuffle_sequence_dict,
    split_tensor_dict,
)
from .distillation_config import DistillationConfig


if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


if is_peft_available():
    import peft
    from peft import PeftConfig, get_peft_model


if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


if is_trackio_available():
    import trackio


if is_wandb_available():
    import wandb


logger = get_logger(__name__)


def _print_completions_sample(prompts: list[str], completions: list[str], step: int, num_samples: int = None) -> None:
    """Print a sample of prompt-completion pairs using rich."""
    if not is_rich_available():
        return

    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")

    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]

    for prompt, completion in zip(prompts, completions, strict=True):
        table.add_row(Text(prompt), Text(completion))
        table.add_section()

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


def _jsd_divergence(student_log_probs, teacher_log_probs, beta, support_mask=None):
    """Compute JSD (or forward/reverse KL) from log-probability tensors.

    When *support_mask* is not None, uses manual computation with masked positions zeroed. When None, uses
    ``F.kl_div``.
    """
    if support_mask is not None:
        safe_student = torch.where(support_mask, student_log_probs, torch.zeros_like(student_log_probs))
        safe_teacher = torch.where(support_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs))
        student_probs = torch.where(support_mask, student_log_probs.exp(), torch.zeros_like(student_log_probs))
        teacher_probs = torch.where(support_mask, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs))

        if beta == 0:
            return torch.nan_to_num(teacher_probs * (safe_teacher - safe_student), nan=0.0)
        elif beta == 1:
            return torch.nan_to_num(student_probs * (safe_student - safe_teacher), nan=0.0)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            tiny = torch.finfo(student_probs.dtype).tiny
            mixture_probs = (1 - beta_t) * student_probs + beta_t * teacher_probs
            safe_mixture = torch.where(
                support_mask,
                torch.log(mixture_probs.clamp_min(tiny)),
                torch.zeros_like(student_log_probs),
            )
            kl_teacher = torch.nan_to_num(teacher_probs * (safe_teacher - safe_mixture), nan=0.0)
            kl_student = torch.nan_to_num(student_probs * (safe_student - safe_mixture), nan=0.0)
            return beta_t * kl_teacher + (1 - beta_t) * kl_student
    else:
        if beta == 0:
            return F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            return F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            return beta_t * kl_teacher + (1 - beta_t) * kl_student


class DistillationTrainer(_BaseTrainer):
    """
    Trainer for knowledge distillation from a teacher model to a student model.

    Supports:
    - Generalized JSD loss (forward KL, reverse KL, or interpolated JSD via `beta`)
    - On-policy distillation: the student generates completions, the teacher scores them
    - Local teacher model
    - Student on-policy generation via vLLM or model.generate()
    - Liger kernel for memory-efficient fused JSD loss
    """

    _tag_names = ["trl", "distillation"]
    _name = "Distillation"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024on-policy,
                title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                year         = 2024,
                booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
                publisher    = {OpenReview.net},
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: DistillationConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        if args is None:
            args = DistillationConfig(output_dir="tmp_distillation")

        # ── Student model loading ──
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model_init_kwargs, str):
            import json

            model_init_kwargs = json.loads(model_init_kwargs)
        teacher_model_init_kwargs = args.teacher_model_init_kwargs or {}
        if isinstance(teacher_model_init_kwargs, str):
            import json

            teacher_model_init_kwargs = json.loads(teacher_model_init_kwargs)
        if isinstance(model, str):
            model_name_or_path = model
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            model_name_or_path = model.config._name_or_path if model is not None else None

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # ── Processing class (tokenizer) ──
        if processing_class is None and model_name_or_path is not None:
            processing_class = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=args.trust_remote_code
            )
        if processing_class is not None:
            if getattr(processing_class, "pad_token", None) is None:
                processing_class.pad_token = processing_class.eos_token
        self._tokenizer = (
            processing_class.tokenizer if isinstance(processing_class, ProcessorMixin) else processing_class
        )

        # ── PEFT ──
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            # ZeRO-3 + PEFT for non-quantized models:
            # - PEFT's default autocast_adapter_dtype=True upcasts LoRA adapter params to fp32 even when the base model is bf16.
            # - ZeRO-3's _allgather_params_coalesced allocates output buffers using the dtype of the first persistent parameter,
            #   so mixed-dtype persistent_parameters (bf16 base + fp32 LoRA) cause a TypeError on the first optimizer step.
            # - Passing autocast_adapter_dtype=False keeps adapter params in the base model dtype (bf16), fixing the mismatch.
            # - This is safe: the fp32 upcast is a QLoRA-specific concern (low-bit quantized base models), not needed for
            #   non-quantized bf16 training.
            # - See:
            #   - TRL issue: https://github.com/huggingface/trl/issues/6089
            #   - Upstream issue: https://github.com/deepspeedai/DeepSpeed/issues/8072
            # - autocast_adapter_dtype was introduced in PEFT 0.12.0; before, no upcast existed: no need to pass the kwarg
            _is_quantized_model = getattr(model, "is_loaded_in_4bit", False) or getattr(
                model, "is_loaded_in_8bit", False
            )
            get_peft_model_kwargs = {}
            if (
                args.deepspeed_plugin is not None
                and args.deepspeed_plugin.zero_stage == 3
                and not _is_quantized_model
                and Version(peft.__version__) >= Version("0.12.0")
            ):
                get_peft_model_kwargs["autocast_adapter_dtype"] = False
            model = get_peft_model(model, peft_config, **get_peft_model_kwargs)

        # ── Liger fused JSD loss ──
        self.use_liger_loss = False
        if args.use_liger_kernel:
            self.liger_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
            )
            self.use_liger_loss = True
            self._forward_redirection = _ForwardRedirection()

        # ── Teacher model setup ──
        # `teacher_model` may be None: subclasses (e.g. ServerDistillationTrainer) supply the teacher another way.
        if teacher_model is not None:
            if isinstance(teacher_model, str):
                teacher_model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
                dtype = teacher_model_init_kwargs.get("dtype")
                teacher_model_init_kwargs["dtype"] = dtype if dtype in ["auto", None] else getattr(torch, dtype)
                if args.teacher_model_revision is not None:
                    teacher_model_init_kwargs.setdefault("revision", args.teacher_model_revision)
                # Distributed training requires device_map=None ("auto" fails)
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    teacher_model_init_kwargs["device_map"] = None
                teacher_model = create_model_from_path(teacher_model, **teacher_model_init_kwargs)
            elif args.teacher_model_init_kwargs is not None:
                raise ValueError(
                    "You passed teacher_model_init_kwargs to the config, but your teacher_model is already "
                    "instantiated."
                )

        # Iterable datasets can't be indexed, so the RepeatSampler can't be attached to them. Instead, the sampler's
        # ordering is reproduced by streaming (see `get_train_dataloader`/`get_eval_dataloader` and
        # `repeat_iterable_dataset`). This requires `dispatch_batches=False`: with the default dispatch path, batches
        # are collated on the main process and Accelerate tries to concatenate the string `prompt` column, which fails;
        # `dispatch_batches=False` also lets each process shard the stream into contiguous slices, as the sampler does.
        # See https://github.com/huggingface/trl/issues/3213
        uses_iterable_dataset = (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        )
        if uses_iterable_dataset:
            if args.accelerator_config.dispatch_batches:
                raise ValueError(
                    "Iterable datasets require `dispatch_batches=False`, but it is set to `True` in "
                    "`accelerator_config`. Please set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False
        # An iterable train set bakes the generation-batch repeats into the stream, so it must be read by a single
        # worker: multiple workers would shard and interleave it, breaking the generation-batch ordering that
        # `_prepare_inputs` relies on. Map-style train keeps its workers.
        if isinstance(train_dataset, IterableDataset) and args.dataloader_num_workers != 0:
            logger.warning(
                f"Iterable datasets require `dataloader_num_workers=0` to preserve prompt grouping; overriding the "
                f"provided value ({args.dataloader_num_workers})."
            )
            args.dataloader_num_workers = 0

        # Trainer does not need to remove unused columns — the collator handles raw data
        args.remove_unused_columns = False

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in distillation
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. Here, loss scaling instead depends on the total number of completion tokens across the global
            # accumulated batch. To control scaling ourselves, we must disable Trainer's built-in scaling. The simplest
            # (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses that behavior
            # without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        self._dist = DistributedBackend(self.accelerator)

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        # ── Prepare teacher model (after super().__init__ so accelerator is ready) ──
        if teacher_model is not None:
            # The divergence compares the full next-token distribution of the student against the teacher's, so both
            # must be defined over the same vocabulary.
            student_vocab_size = self.model.config.get_text_config().vocab_size
            teacher_vocab_size = teacher_model.config.get_text_config().vocab_size
            if student_vocab_size != teacher_vocab_size:
                raise ValueError(
                    f"The student model has vocab_size {student_vocab_size} but the teacher model has vocab_size "
                    f"{teacher_vocab_size}. Distillation compares the teacher's full next-token distribution, which "
                    f"requires a shared vocabulary. Use a teacher with the same vocab_size, or GOLD for "
                    f"cross-tokenizer distillation."
                )
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        else:
            self.teacher_model = None

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        # ── Store config values ──
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.pad_to_multiple_of = args.pad_to_multiple_of
        self.shuffle_dataset = args.shuffle_dataset

        self._step = 0
        self._buffered_inputs = None

        # Ensure each process receives a unique seed so different processes generate different completions when
        # generating with transformers. We could skip it if we use vLLM, but it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # ── Generation config ──
        generation_kwargs = {
            "max_new_tokens": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": True,
            "top_k": args.top_k,
            "pad_token_id": self.processing_class.pad_token_id,
        }
        self.generation_config = GenerationConfig(**generation_kwargs)
        self.generation_kwargs = generation_kwargs
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # ── Metrics & Logging ──
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completions_steps = args.log_completions_steps
        self.num_completions_to_print = args.num_completions_to_print

        self._textual_logs = {
            "prompt": [],
            "completion": [],
        }

        # ── vLLM for student generation ──
        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                processing_class=self.processing_class,
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size * args.gradient_accumulation_steps,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                trust_remote_code=args.trust_remote_code,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_completion_length=args.max_completion_length,
                logprobs=None,
            )
            self._last_loaded_step = -1

    # ──────────────────────────────────────────────────────────────────────
    #  Dataset / Dataloader
    # ──────────────────────────────────────────────────────────────────────

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In DistillationTrainer, we preprocess data, so using the model's signature columns
        # doesn't work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    def get_train_dataloader(self):
        dataset = self.train_dataset
        if isinstance(dataset, IterableDataset):
            # Iterable datasets can't be indexed, so RepeatSampler can't be attached. Reproduce its ordering by
            # transforming the stream instead (see `repeat_iterable_dataset`). The full permutation done by
            # RepeatSampler becomes a buffered shuffle here.
            if self.shuffle_dataset:
                dataset = dataset.shuffle(seed=self.args.seed)
            dataset = repeat_iterable_dataset(
                dataset,
                mini_repeat_count=1,
                batch_size=self.args.generation_batch_size,
                repeat_count=self.args.gradient_accumulation_steps,
            )
        return self._get_dataloader(
            dataset=dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Repeat each generation batch `gradient_accumulation_steps` times so the completions generated once per
        # generation batch (see `_prepare_inputs`) are reused across the accumulation window. Distillation is n=1,
        # so there is no per-prompt repeat (`mini_repeat_count=1`).
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=1,
            batch_size=self.args.generation_batch_size,
            repeat_count=self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            seed=self.args.seed,
        )

    # This method overrides `Trainer.get_eval_dataloader` to wrap iterable eval datasets, reproducing the
    # RepeatSampler ordering that can't be attached to them (see `get_train_dataloader`). Map-style datasets keep the
    # default path via `_get_eval_sampler`, which shuffles with `seed`, so the iterable wrap shuffles too (buffered)
    # to walk prompts in a matching order.
    # Maintenance note: this method is a copy-paste of the original `Trainer.get_eval_dataloader`, with the iterable
    # wrapping as the only addition.
    def get_eval_dataloader(self, eval_dataset: str | Dataset | IterableDataset | None = None) -> DataLoader:
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
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        if isinstance(eval_dataset, IterableDataset):
            # Apply the `__init__` iterable config here too
            if self.args.accelerator_config.dispatch_batches:
                raise ValueError(
                    "Iterable datasets require `dispatch_batches=False`, but it is set to `True` in "
                    "`accelerator_config`. Please set it to `False`."
                )
            self.accelerator.dataloader_config.dispatch_batches = False
            eval_dataset = eval_dataset.shuffle(seed=self.args.seed)
            eval_dataset = repeat_iterable_dataset(eval_dataset, mini_repeat_count=1)
            # Force a single worker for this loader only, without persisting the change
            num_workers = self.args.dataloader_num_workers
            self.args.dataloader_num_workers = 0

        try:
            return self._get_dataloader(
                dataset=eval_dataset,
                description="Evaluation",
                batch_size=self.args.eval_batch_size,
                sampler_fn=self._get_eval_sampler,
                dataloader_key=dataloader_key,
            )
        finally:
            if isinstance(eval_dataset, IterableDataset):
                self.args.dataloader_num_workers = num_workers

    def _tokenize_prompts(self, prompts: list):
        """Tokenize prompts and extract multimodal fields for generation.

        Conversational prompts (a list of chat messages) are rendered with the chat template and a trailing generation
        prompt; standard prompts (plain strings) are tokenized directly. The per-example tools/environments path and
        the image extraction are added with VLM support later (issue #6449). Unwired until the GRPO generation stack
        lands.
        """
        if is_conversational({"prompt": prompts[0]}):
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.chat_template_kwargs,
            )
            prompt_ids = tokenized["input_ids"]
            # For VLMs, the processor returns extra multimodal fields (pixel_values, image_grid_thw, etc.)
            multimodal_fields = {k: v for k, v in tokenized.items() if k not in ("input_ids", "attention_mask")}
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
            multimodal_fields = {}
        images = None  # extracted from the messages once VLM support lands
        return prompt_ids, images, multimodal_fields

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        device = self.accelerator.device

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # Sync weights if training step changed
            if self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Generate using vLLM with raw token IDs. Distillation is n=1 and uses the teacher distribution rather than
            # sampled logprobs, so we request one completion per prompt and discard vLLM's logprobs.
            _, completion_ids, _, _ = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=images,
                num_generations=1,
                profiler=profiling_context(self, "vLLM.generate"),
            )

        else:
            # Regular generation path: left-pad token IDs into tensors
            prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
            padded_ids = pad(prompt_tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left")
            attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left")
            generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}
            # For VLMs, include multimodal fields as tensors (pixel_values, image_grid_thw, etc.)
            for k, v in multimodal_fields.items():
                if isinstance(v, torch.Tensor):
                    generate_inputs[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], list):
                    # Per-token field (e.g., token_type_ids): left-pad like input_ids
                    generate_inputs[k] = pad([torch.tensor(x) for x in v], padding_value=0, padding_side="left")
                else:
                    generate_inputs[k] = torch.tensor(np.array(v))
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model,
                torch.no_grad(),
                self._dist.summon_full_params(self.model_wrapped, recurse=False),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config
                )
            # Compute prompt length and extract completion ids
            prompt_length = generate_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self._tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids = [
                c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
            ]

        return completion_ids

    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        prompt_ids, images, multimodal_fields = self._tokenize_prompts(prompts)
        completion_ids = self._generate_single_turn(prompt_ids, images, multimodal_fields)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + agg_completion_lengths.sum()).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids

    # Name kept aligned with GRPO/RLOO for consistency; distillation has no rewards, so nothing is actually scored.
    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | Any]]) -> dict[str, Any]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]

        prompt_ids_list, completion_ids_list = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)

        num_items_in_batch = self.accelerator.gather(completion_mask.sum()).sum()

        # Log the prompt and completion texts
        if self.log_completions:
            prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            self._textual_logs["prompt"].extend(gather_object(prompts_text))
            self._textual_logs["completion"].extend(gather_object(completions_text))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": num_items_in_batch,
        }
        return output

    # ──────────────────────────────────────────────────────────────────────
    #  Buffering across gradient accumulation steps
    # ──────────────────────────────────────────────────────────────────────

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × gradient accumulation steps)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.gradient_accumulation_steps)
                self._buffered_inputs = generation_batches
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    # ──────────────────────────────────────────────────────────────────────
    #  Loss computation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reduce_divergence_loss(jsd, completion_mask=None, reduction="batchmean", num_items_in_batch=None):
        """Reduce a per-token divergence tensor over the valid completion tokens.

        When `num_items_in_batch` is provided (as under gradient accumulation), the divergence is reduced as `sum /
        num_items_in_batch`, matching the gradient-accumulation-correct behavior of HF's default cross-entropy.
        Otherwise it falls back to the local `reduction` (default `batchmean`). See issue #4719.
        """
        mask = None
        if completion_mask is not None:
            mask = completion_mask.bool()
            jsd = jsd[mask]

        if num_items_in_batch is not None:
            # Normalize by the global number of valid tokens for gradient-accumulation-correct loss.
            jsd_sum = jsd.sum()
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(jsd_sum.device)
            return jsd_sum / num_items_in_batch
        if reduction == "batchmean":
            # clamp_min(1) avoids 0/0 -> nan when a sample has no unmasked positions
            # (e.g. completion fully truncated). jsd[mask] is empty -> jsd.sum() == 0,
            # so 0/1 == 0 with a valid grad path.
            denom = mask.sum().clamp_min(1) if completion_mask is not None else max(jsd.size(0), 1)
            return jsd.sum() / denom
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        num_items_in_batch=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation over the full vocabulary.

        Args:
            student_logits: Tensor of shape (batch_size, sequence_length, vocab_size).
            teacher_logits: Tensor of shape (batch_size, sequence_length, vocab_size).
            labels: Tensor of shape (batch_size, sequence_length) with -100 for positions to ignore.
            beta: Interpolation coefficient. 0.0 = forward KL, 1.0 = reverse KL.
            temperature: Softmax temperature.
            reduction: 'batchmean', 'sum', 'mean', or 'none'.

        Returns:
            Scalar loss tensor.
        """
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        jsd = _jsd_divergence(student_log_probs, teacher_log_probs, beta)
        return DistillationTrainer._reduce_divergence_loss(
            jsd,
            completion_mask=(labels != -100) if labels is not None else None,
            reduction=reduction,
            num_items_in_batch=num_items_in_batch,
        )

    def _get_teacher_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get logits from the local teacher model."""
        if self.teacher_model is None:
            raise ValueError("No teacher model configured.")
        self.teacher_model.eval()
        with torch.no_grad():
            return self.teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # transformers computes `num_items_in_batch` from the raw dataloader labels, before on-policy generation
        # replaces the completions; use the count over the generated completions instead (computed in
        # `_generate_and_score_completions`). Divide by the process count so the per-process loss compensates for DDP
        # gradient averaging (as GRPO does).
        if self.model.training and inputs.get("num_items_in_batch") is not None:
            num_items_in_batch = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes

        if self.use_liger_loss:
            loss = self._compute_liger_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            return (loss, None) if return_outputs else loss

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # only the completion tokens are trained on

        # Student forward pass
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        teacher_logits = self._get_teacher_logits(input_ids, attention_mask)
        student_logits = student_outputs.logits[:, -logits_to_keep - 1 : -1, :]
        teacher_logits = teacher_logits[:, -logits_to_keep - 1 : -1, :]
        jsd = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            beta=self.beta,
            temperature=self.temperature,
            reduction="none",
        )
        loss = self._reduce_divergence_loss(
            jsd, completion_mask=completion_mask, num_items_in_batch=num_items_in_batch
        )

        return (loss, student_outputs) if return_outputs else loss

    def _liger_student_forward(self, student, inputs):
        """Decoder-only forward used by the Liger JSD path (skips lm_head to save memory)."""
        if hasattr(student, "get_decoder") and student.get_decoder() is not None:
            decoder = student.get_decoder()
        else:
            decoder = getattr(student, getattr(student, "base_model_prefix", "model"), student)
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        return decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    def _compute_liger_loss(self, model, inputs, num_items_in_batch=None):
        """Memory-efficient JSD using Liger kernel (operates on hidden states, not full logits)."""
        # Route through the DDP/FSDP wrapper via _forward_redirection so that
        # DDP.forward() is called and prepare_for_backward() fires correctly.
        unwrapped_student = self.accelerator.unwrap_model(model)
        student_outputs = self._forward_redirection(
            model, unwrapped_student, self._liger_student_forward, unwrapped_student, inputs
        )

        self.teacher_model.eval()
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        if hasattr(unwrapped_teacher, "get_decoder") and unwrapped_teacher.get_decoder() is not None:
            base_teacher = unwrapped_teacher.get_decoder()
        else:
            base_teacher = getattr(
                unwrapped_teacher, getattr(unwrapped_teacher, "base_model_prefix", "model"), unwrapped_teacher
            )
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        with torch.no_grad():
            teacher_outputs = base_teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        student_hidden = student_outputs.last_hidden_state[:, :-1]
        teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]
        del student_outputs, teacher_outputs

        student_hidden = student_hidden.reshape(-1, student_hidden.shape[-1])
        teacher_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])

        completion_mask = torch.cat([torch.zeros_like(inputs["prompt_mask"]), inputs["completion_mask"]], dim=1).bool()
        masked_input_ids = torch.where(completion_mask, input_ids, torch.full_like(input_ids, -100))
        true_labels = masked_input_ids[:, 1:].reshape(-1)

        student_head = unwrapped_student.get_output_embeddings()
        teacher_head = unwrapped_teacher.get_output_embeddings()

        loss = self.liger_loss(
            student_input=student_hidden,
            student_weight=student_head.weight,
            teacher_input=teacher_hidden,
            teacher_weight=teacher_head.weight,
            true_labels=true_labels,
            student_bias=getattr(student_head, "bias", None),
            teacher_bias=getattr(teacher_head, "bias", None),
        )

        # The Liger JSD loss normalizes by the local number of valid tokens. Under gradient accumulation we want
        # the global normalization, so rescale by `num_valid_local / num_items_in_batch`.
        if num_items_in_batch is not None:
            num_valid_local = (true_labels != -100).sum().clamp_min(1)
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss * num_valid_local / num_items_in_batch

        del student_hidden, teacher_hidden, true_labels
        return loss

    def _get_liger_zero3_lm_head_gather_ctx(self, model: nn.Module):
        """Context manager for gathering lm_head parameters under Liger + ZeRO-3."""
        if not self.use_liger_loss:
            return nullcontext()

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is None or deepspeed_plugin.zero_stage != 3:
            return nullcontext()

        import deepspeed

        unwrapped_student = self.accelerator.unwrap_model(model)
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        student_head = unwrapped_student.get_output_embeddings()
        teacher_head = unwrapped_teacher.get_output_embeddings()
        params = [student_head.weight, teacher_head.weight]
        if student_head.bias is not None:
            params.append(student_head.bias)
        if teacher_head.bias is not None:
            params.append(teacher_head.bias)
        return deepspeed.zero.GatheredParameters(params, modifier_rank=None)

    # ──────────────────────────────────────────────────────────────────────
    #  Training step & Logging
    # ──────────────────────────────────────────────────────────────────────

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        with self._get_liger_zero3_lm_head_gather_ctx(model):
            output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

        # Log completions to console, wandb, and trackio
        should_log_completions = (
            self.log_completions
            and self.state.global_step > 0
            and self.state.global_step % self.log_completions_steps == 0
        )

        if should_log_completions and self.accelerator.is_main_process:
            prompts = list(self._textual_logs["prompt"])
            completions = list(self._textual_logs["completion"])

            if prompts:
                _print_completions_sample(prompts, completions, self.state.global_step, self.num_completions_to_print)

                logging_backends = []
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    logging_backends.append(wandb)
                if self.args.report_to and "trackio" in self.args.report_to:
                    logging_backends.append(trackio)

                if logging_backends:
                    import pandas as pd

                    table_data = {
                        "step": [str(self.state.global_step)] * len(prompts),
                        "prompt": prompts,
                        "completion": completions,
                    }
                    df = pd.DataFrame(table_data)
                    if self.num_completions_to_print and len(df) > self.num_completions_to_print:
                        df = df.sample(n=self.num_completions_to_print, random_state=42)

                    for logging_backend in logging_backends:
                        logging_backend.log({"completions": logging_backend.Table(dataframe=df)})

        # Clear text logs on all processes after the logging interval
        if should_log_completions:
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
