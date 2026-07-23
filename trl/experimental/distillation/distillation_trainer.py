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

import random
import textwrap
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather_object
from datasets import Dataset
from packaging.version import Version
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainerCallback, is_trackio_available, is_wandb_available
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.utils import is_liger_kernel_available, is_peft_available, is_rich_available

from ...data_utils import is_conversational
from ...extras.profiling import profiling_decorator
from ...generation.vllm_generation import VLLMGeneration
from ...import_utils import is_vllm_available
from ...models import prepare_deepspeed
from ...models.utils import _ForwardRedirection, unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import RepeatSampler, create_model_from_path, disable_dropout_in_model, pad, split_tensor_dict
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


class _DistillationCollator:
    """Data collator for the distillation trainer.

    Accepts a prompt-only dataset (a ``prompt`` column, the format shared with GRPO): the student generates the
    completion on-policy, so there is nothing to train on in the dataset.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        all_prompt_ids: list[list[int]] = []

        for example in examples:
            formatted_prompt = self.tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(formatted_prompt, padding=False, add_special_tokens=False)["input_ids"]

            # Prompt-only: no completion to train on (on-policy will generate one)
            all_input_ids.append(list(prompt_ids))
            all_labels.append([self.ignore_index] * len(prompt_ids))
            all_prompt_ids.append(list(prompt_ids))

        # Convert to tensors and left-pad
        pad_id = self.tokenizer.pad_token_id
        input_ids_t = pad(
            [torch.tensor(ids, dtype=torch.long) for ids in all_input_ids],
            padding_side="left",
            padding_value=pad_id,
        )
        attention_mask_t = pad(
            [torch.ones(len(ids), dtype=torch.long) for ids in all_input_ids],
            padding_side="left",
            padding_value=0,
        )
        labels_t = pad(
            [torch.tensor(lab, dtype=torch.long) for lab in all_labels],
            padding_side="left",
            padding_value=self.ignore_index,
        )
        prompts_t = pad(
            [torch.tensor(ids, dtype=torch.long) for ids in all_prompt_ids],
            padding_side="left",
            padding_value=pad_id,
        )
        prompt_mask_t = pad(
            [torch.ones(len(ids), dtype=torch.long) for ids in all_prompt_ids],
            padding_side="left",
            padding_value=0,
        )

        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "labels": labels_t,
            "prompts": prompts_t,
            "prompt_attention_mask": prompt_mask_t,
            "prompt_ids": prompts_t,
            "prompt_mask": prompt_mask_t,
            "completion_ids": input_ids_t[:, prompts_t.shape[1] :],
            "completion_mask": torch.ones_like(input_ids_t[:, prompts_t.shape[1] :]),
        }


class _RepeatBatchDataLoader:
    """Repeats each collated batch ``repeat_count`` times without re-collation.

    ``RepeatSampler`` with ``repeat_count > 1`` causes the DataLoader to re-collate (re-tokenize) the same examples on
    every repeat, which is wasteful. This wrapper instead keeps ``repeat_count=1`` in the sampler and repeats the
    already-collated tensor dict, avoiding redundant tokenization.
    """

    def __init__(self, dataloader, repeat_count: int):
        self.dataloader = dataloader
        self.repeat_count = repeat_count

    def __iter__(self):
        for batch in self.dataloader:
            for _ in range(self.repeat_count):
                yield batch

    def __len__(self):
        return len(self.dataloader) * self.repeat_count

    def set_epoch(self, epoch: int):
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(epoch)

    def __getattr__(self, attr):
        return getattr(self.dataloader, attr)


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

        # ── Processing class (tokenizer) ──
        if processing_class is None and model_name_or_path is not None:
            processing_class = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=args.trust_remote_code
            )
        if processing_class is not None:
            if getattr(processing_class, "pad_token", None) is None:
                processing_class.pad_token = processing_class.eos_token

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

        # ── Data collator ──
        if data_collator is None:
            data_collator = _DistillationCollator(tokenizer=processing_class)

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

        # Trainer does not need to remove unused columns — the collator handles raw data
        args.remove_unused_columns = False

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
        self.num_generations = args.num_generations

        # ── Buffer state ──
        self._buffered_inputs = None
        self._buffered_text_logs = None
        self._buffered_num_items = None
        self._buffer_step = 0

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
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

    def _get_completion_lengths(self, generated_tokens: torch.Tensor, prompt_width: int) -> torch.Tensor:
        """Infer per-sample completion lengths from generated tokens."""
        completion_tokens = generated_tokens[:, prompt_width:]
        pad_token_id = self.processing_class.pad_token_id
        eos_token_id = self.generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_ids = set()
        elif isinstance(eos_token_id, int):
            eos_token_ids = {eos_token_id}
        else:
            eos_token_ids = set(eos_token_id)
        pad_equals_eos = pad_token_id is not None and pad_token_id in eos_token_ids

        completion_lengths = []
        for row in completion_tokens.tolist():
            if pad_equals_eos and eos_token_ids:
                completion_length = len(row)
                for idx, token_id in enumerate(row):
                    if token_id in eos_token_ids:
                        completion_length = idx + 1
                        break
            elif pad_token_id is not None:
                completion_length = len(row)
                while completion_length > 0 and row[completion_length - 1] == pad_token_id:
                    completion_length -= 1
            else:
                completion_length = len(row)
            completion_lengths.append(completion_length)

        return torch.tensor(completion_lengths, device=generated_tokens.device, dtype=torch.long)

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

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size * self.accelerator.num_processes,
            repeat_count=1,
            shuffle=True,
            seed=self.args.seed,
        )

    def get_train_dataloader(self):
        """
        Override to load one generation batch per optimizer window.

        The dataloader yields batches of size `per_device_train_batch_size * gradient_accumulation_steps`.
        RepeatSampler ensures each generation batch is repeated `gradient_accumulation_steps` times so the Trainer's
        loop iterates the correct number of times.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
            if self.args.dataloader_num_workers > 0:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        base_dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return _RepeatBatchDataLoader(base_dataloader, repeat_count=self.args.gradient_accumulation_steps)

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
            )
            prompt_ids = tokenized["input_ids"]
            # For VLMs, the processor returns extra multimodal fields (pixel_values, image_grid_thw, etc.)
            multimodal_fields = {k: v for k, v in tokenized.items() if k not in ("input_ids", "attention_mask")}
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
            multimodal_fields = {}
        images = None  # extracted from the messages once VLM support lands
        return prompt_ids, images, multimodal_fields

    # ──────────────────────────────────────────────────────────────────────
    #  Buffering across gradient accumulation steps
    # ──────────────────────────────────────────────────────────────────────

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        if not self.model.training:
            return generation_batch

        buffer_steps = self.args.gradient_accumulation_steps
        if self._buffer_step % buffer_steps == 0 or self._buffered_inputs is None:
            self._fill_buffer(generation_batch, buffer_steps)

        slice_idx = self._buffer_step % buffer_steps
        inputs = self._buffered_inputs[slice_idx]
        self._buffer_step += 1
        return inputs

    @profiling_decorator
    def _fill_buffer(self, generation_batch: dict[str, torch.Tensor | Any], buffer_steps: int):
        """Split the batch into slices and generate student completions for each (always on-policy)."""
        slices = split_tensor_dict(generation_batch, buffer_steps)

        self._buffered_inputs = [None] * buffer_steps
        self._buffered_text_logs = [None] * buffer_steps

        # Generate student completions for every slice
        self._generate_student_completions(slices, list(range(buffer_steps)))

        # Loss denominator (`num_items_in_batch`): the global number of completion tokens actually trained on this
        # optimizer step. transformers derives its own count from the *raw dataloader* labels — before generation
        # replaces the completions — which is wrong for on-policy training and zero for prompt-only datasets (dividing
        # the loss by zero). Recompute it here from the generated labels, gathered across processes (issue #4719).
        local_completion_tokens = sum(int(s["completion_mask"].sum()) for s in self._buffered_inputs if s is not None)
        self._buffered_num_items = self.accelerator.gather(
            torch.tensor(local_completion_tokens, device=self.accelerator.device)
        ).sum()

        # Gather text logs once per optimizer step (all processes must participate)
        if self.log_completions:
            prompts_all = []
            completions_all = []
            for entry in self._buffered_text_logs:
                if entry is not None:
                    prompts, completions = entry
                    prompts_all.extend(prompts)
                    completions_all.extend(completions)
            self._textual_logs["prompt"].extend(gather_object(prompts_all))
            self._textual_logs["completion"].extend(gather_object(completions_all))

    @profiling_decorator
    def _generate_student_completions(self, slices: list[dict[str, torch.Tensor | Any]], on_policy_indices: list[int]):
        """Generate completions from the student model for on-policy training."""
        if not self.use_vllm:
            self._generate_with_model(slices, on_policy_indices)
            return

        # Collect all prompts across on-policy slices, stripping left-padding so vLLM
        # receives only real prompt token IDs (like GRPO trainer).
        local_prompts = []
        local_slice_indices = []
        pad_token_id = self.processing_class.pad_token_id
        for slice_idx in on_policy_indices:
            prompt_mask = slices[slice_idx].get("prompt_attention_mask")
            for i, prompt in enumerate(slices[slice_idx]["prompts"]):
                if prompt_mask is not None:
                    prompt = prompt[prompt_mask[i].bool()]
                elif pad_token_id is not None:
                    first_non_pad = (prompt != pad_token_id).nonzero(as_tuple=True)[0]
                    if len(first_non_pad) > 0:
                        prompt = prompt[first_non_pad[0] :]
                local_prompts.append(prompt)
                local_slice_indices.append(slice_idx)

        # Sync student weights to vLLM if needed
        if (
            self.state.global_step != self._last_vllm_sync_step
            and self.state.global_step % self.vllm_sync_frequency == 0
        ):
            self.vllm_generation.sync_weights()
            self._last_vllm_sync_step = self.state.global_step

        # Generate completions — pass token IDs directly, no text decoding
        prompt_ids_list = [p.tolist() for p in local_prompts]
        _, completion_ids, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids_list, images=None, num_generations=self.num_generations
        )

        # Process completions into the buffer
        self._store_completions_in_buffer(
            slices, on_policy_indices, local_slice_indices, local_prompts, completion_ids
        )

    def _generate_with_model(self, slices: list[dict[str, torch.Tensor | Any]], on_policy_indices: list[int]):
        """Fallback generation using model.generate() (no vLLM)."""
        with unwrap_model_for_generation(
            self.model, self.accelerator, generation_kwargs=self.generation_kwargs
        ) as unwrapped_model:
            for slice_idx in on_policy_indices:
                slice_inputs = slices[slice_idx]
                generated_outputs = unwrapped_model.generate(
                    input_ids=slice_inputs["prompts"],
                    attention_mask=slice_inputs.get("prompt_attention_mask", None),
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                )
                generated_tokens = generated_outputs.sequences
                batch_size = generated_tokens.size(0)
                device = generated_tokens.device
                pad_token_id = self.processing_class.pad_token_id
                prompt_width = slice_inputs["prompts"].shape[1]
                prompt_mask = slice_inputs.get("prompt_attention_mask")
                if prompt_mask is not None:
                    prompt_token_lengths = prompt_mask.sum(dim=1)
                else:
                    prompt_token_lengths = torch.full((batch_size,), prompt_width, dtype=torch.long, device=device)
                completion_lengths = self._get_completion_lengths(generated_tokens, prompt_width)
                new_attention_mask, new_labels, new_completion_mask = self._build_sequence_batch(
                    generated_tokens, prompt_width, prompt_token_lengths, completion_lengths
                )

                # Decode for logging
                prompt_texts = []
                completion_texts = []
                for idx in range(batch_size):
                    prompt_tokens = slice_inputs["prompts"][idx]
                    if prompt_mask is not None:
                        prompt_tokens = prompt_tokens[prompt_mask[idx].bool()]
                    elif pad_token_id is not None:
                        prompt_tokens = prompt_tokens[prompt_tokens != pad_token_id]
                    prompt_texts.append(
                        self.processing_class.decode(prompt_tokens.tolist(), skip_special_tokens=False)
                    )
                    length = prompt_width
                    completion_length = int(completion_lengths[idx].item())
                    completion_texts.append(
                        self.processing_class.decode(
                            generated_tokens[idx, length : length + completion_length].tolist(),
                            skip_special_tokens=False,
                        )
                    )

                updated = dict(slice_inputs)
                updated["input_ids"] = generated_tokens
                updated["attention_mask"] = new_attention_mask
                updated["labels"] = new_labels
                updated["completion_mask"] = new_completion_mask
                updated["prompt_ids"] = slice_inputs["prompts"]
                updated["prompt_mask"] = prompt_mask
                updated["completion_ids"] = generated_tokens[:, prompt_width:]

                self._buffered_inputs[slice_idx] = updated
                self._buffered_text_logs[slice_idx] = (prompt_texts, completion_texts)

    def _store_completions_in_buffer(
        self,
        slices: list[dict[str, torch.Tensor | Any]],
        on_policy_indices: list[int],
        local_slice_indices: list[int],
        local_prompts: list[torch.Tensor],
        completion_ids: list,
    ):
        """Process vLLM completions and store them in the buffer.

        Uses original prompt token IDs directly (no decode/re-encode roundtrip), following the same approach as
        GRPOTrainer.
        """
        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
        max_completion_length = self.generation_config.max_new_tokens

        # Group completions and prompt token IDs by slice
        slice_completions = {idx: [] for idx in on_policy_indices}
        slice_prompt_ids = {idx: [] for idx in on_policy_indices}
        for i, slice_idx in enumerate(local_slice_indices):
            slice_completions[slice_idx].append(completion_ids[i])
            slice_prompt_ids[slice_idx].append(local_prompts[i])

        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            prompt_id_tensors = slice_prompt_ids[slice_idx]
            prompt_width = max(len(p) for p in prompt_id_tensors)
            prompt_token_lengths = torch.tensor([len(p) for p in prompt_id_tensors], device=device, dtype=torch.long)
            prompt_attention_mask = (
                torch.arange(prompt_width, device=device).unsqueeze(0)
                >= (prompt_width - prompt_token_lengths).unsqueeze(1)
            ).long()

            # Left-pad prompt token IDs to the longest prompt in this slice
            prompt_ids = torch.stack(
                [F.pad(p, (prompt_width - len(p), 0), value=pad_token_id) for p in prompt_id_tensors]
            ).to(device)

            # Pad/truncate completions (right-pad to max_completion_length)
            completion_tensors = []
            completion_ids_for_text = []
            completion_lengths = []
            for comp_ids in slice_completions[slice_idx]:
                t = torch.tensor(comp_ids, device=device)
                if len(t) > max_completion_length:
                    t = t[:max_completion_length]
                completion_ids_for_text.append(t.tolist())
                completion_lengths.append(len(t))
                if len(t) < max_completion_length:
                    padding = torch.full((max_completion_length - len(t),), pad_token_id, device=device, dtype=t.dtype)
                    t = torch.cat([t, padding])
                completion_tensors.append(t)

            completion_ids_padded = torch.stack(completion_tensors)
            new_input_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
            completion_lengths = torch.tensor(completion_lengths, device=device, dtype=torch.long)
            new_attention_mask, new_labels, new_completion_mask = self._build_sequence_batch(
                new_input_ids, prompt_width, prompt_token_lengths, completion_lengths
            )

            # Decode for logging only
            prompt_texts = self.processing_class.batch_decode(
                prompt_id_tensors, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            completion_texts = self.processing_class.batch_decode(
                completion_ids_for_text, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

            updated = dict(slice_inputs)
            updated["input_ids"] = new_input_ids
            updated["attention_mask"] = new_attention_mask
            updated["labels"] = new_labels
            updated["completion_mask"] = new_completion_mask
            updated["prompt_ids"] = prompt_ids
            updated["prompt_mask"] = prompt_attention_mask
            updated["completion_ids"] = completion_ids_padded
            # Update prompts to match the new padding width so prompt_length is consistent
            updated["prompts"] = prompt_ids
            updated["prompt_attention_mask"] = prompt_attention_mask

            self._buffered_inputs[slice_idx] = updated
            self._buffered_text_logs[slice_idx] = (prompt_texts, completion_texts)

    @staticmethod
    def _build_sequence_batch(
        new_input_ids: torch.Tensor,
        prompt_width: int,
        prompt_token_lengths: torch.Tensor,
        completion_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build attention mask, labels, and completion mask from prompt/completion lengths."""
        prompt_token_lengths = prompt_token_lengths.to(device=new_input_ids.device, dtype=torch.long)
        completion_lengths = completion_lengths.to(device=new_input_ids.device, dtype=torch.long)
        positions = torch.arange(new_input_ids.shape[1], device=new_input_ids.device).unsqueeze(0)
        prompt_mask = (positions < prompt_width) & (positions >= (prompt_width - prompt_token_lengths).unsqueeze(1))
        completion_mask = (positions >= prompt_width) & (positions < (prompt_width + completion_lengths).unsqueeze(1))
        new_attention_mask = (prompt_mask | completion_mask).long()

        new_labels = torch.full_like(new_input_ids, -100)
        new_labels[completion_mask] = new_input_ids[completion_mask]

        # Region-shaped completion mask (B, completion_width), aligned with `completion_ids`, as GRPO emits it.
        return new_attention_mask, new_labels, completion_mask[:, prompt_width:].long()

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
        # replaces the completions; use the count over the generated completions instead (computed in `_fill_buffer`).
        # Divide by the process count so the per-process loss compensates for DDP gradient averaging (as GRPO does).
        if self.model.training and self._buffered_num_items is not None:
            num_items_in_batch = self._buffered_num_items.clamp(min=1.0) / self.accelerator.num_processes

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
        """Training step: generate on-policy, then run the parent step, tracking completion stats."""
        buffer_steps = self.args.gradient_accumulation_steps

        with self._get_liger_zero3_lm_head_gather_ctx(model):
            loss = super().training_step(model, inputs, num_items_in_batch)

        slice_idx = (self._buffer_step - 1) % buffer_steps

        # Track completion length stats — read from buffered inputs (which reflect the generated completions)
        actual_inputs = self._buffered_inputs[slice_idx] if self._buffered_inputs is not None else inputs
        labels = actual_inputs.get("labels")
        if labels is not None:
            completion_lengths = (labels != -100).sum(dim=1).float()
            gathered_lengths = self.accelerator.gather(completion_lengths)
            mode = "train"
            self._metrics[mode]["completions/mean_length"].append(gathered_lengths.mean().item())
            self._metrics[mode]["completions/max_length"].append(gathered_lengths.max().item())
            self._metrics[mode]["completions/min_length"].append(gathered_lengths.min().item())

            # Log fraction of completions that hit max_completion_length (truncated)
            max_comp_len = getattr(self.generation_config, "max_new_tokens", None)
            if max_comp_len is not None:
                truncated_frac = (gathered_lengths >= max_comp_len).float().mean().item()
                self._metrics[mode]["completions/truncated_fraction"].append(truncated_frac)

        return loss

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
